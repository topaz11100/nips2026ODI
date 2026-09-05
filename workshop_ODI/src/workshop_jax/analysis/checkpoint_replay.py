from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from workshop_jax.analysis.spectral import dft_magnitude_sum, layer_mean_distance
from workshop_jax.artifacts import (
    feature_artifact_path,
    load_yaml,
    remove_tree,
    restore_model_checkpoint,
    write_csv,
    write_yaml,
)
from workshop_jax.config import AnalysisConfig
from workshop_jax.data.dataset import PreparedDataset, index_batches
from workshop_jax.models import initialize, signal_apply
from workshop_jax.paper_spec import build_model_plan, experiment_spec


D_MODEL_COLUMNS = (
    "dataset",
    "seed",
    "epoch",
    "train_d_model",
    "test_d_model",
)


def _restored_model_state(base: Mapping[str, Any], snapshot: Mapping[str, Any]) -> Any:
    state = dict(base)
    saved = snapshot.get("model_state")
    if isinstance(saved, Mapping):
        state.update(saved)
    return state


def _spectrum_kernel(plan: Any) -> Any:
    def kernel(
        params: Any,
        model_state: Any,
        inputs: Any,
        mask: Any,
    ) -> tuple[tuple[Any, Any], ...]:
        applied = signal_apply(
            params,
            model_state,
            inputs,
            plan=plan,
        )
        spike_traces = tuple(applied.traces["spike_output"])
        if len(spike_traces) != len(plan.hidden_sizes) + 1:
            raise ValueError("signal trace count does not match the model plan")
        hidden = spike_traces[:-1]
        return tuple(dft_magnitude_sum(signal, mask) for signal in hidden)

    return jax.jit(kernel)


def _layer_spectra(
    dataset: PreparedDataset,
    split: str,
    *,
    batch_size: int,
    kernel: Any,
    params: Any,
    model_state: Any,
) -> tuple[np.ndarray, ...]:
    totals: list[np.ndarray] | None = None
    counts: list[np.float64] | None = None
    for selected in index_batches(dataset.probe_indices(split), batch_size):
        host = dataset.read_batch(split, selected, batch_size=batch_size)
        batch_values = kernel(
            params,
            model_state,
            jax.device_put(host.inputs),
            jax.device_put(host.real_sample_mask),
        )
        host_values = jax.device_get(batch_values)
        if totals is None:
            totals = [
                np.asarray(total, dtype=np.float64).copy() for total, _ in host_values
            ]
            counts = [np.float64(count) for _, count in host_values]
        else:
            if counts is None:
                raise RuntimeError("layer spectrum counts were not initialized")
            for layer_index, (total, count) in enumerate(host_values):
                np.add(
                    totals[layer_index],
                    np.asarray(total, dtype=np.float64),
                    out=totals[layer_index],
                )
                counts[layer_index] = np.float64(
                    counts[layer_index] + np.float64(count)
                )
    if totals is None or counts is None:
        raise ValueError(f"empty probe for {dataset.dataset_id}/{split}")
    if any(count <= 0.0 for count in counts):
        raise ValueError(f"empty probe for {dataset.dataset_id}/{split}")
    return tuple(total / count for total, count in zip(totals, counts, strict=True))


def _reference_spectrum(
    dataset_name: str,
    split: str,
    artifact_root: Path,
) -> np.ndarray:
    with np.load(
        feature_artifact_path(artifact_root, "dataset_signal_analysis")
        / dataset_name
        / "reference_dft.npz",
        allow_pickle=False,
    ) as reference_file:
        return np.asarray(reference_file[f"{split}_spectrum"], dtype=np.float64)


def _distance(reference: np.ndarray, layers: Sequence[np.ndarray]) -> float:
    if not layers:
        raise ValueError("D_model requires at least one estimate layer")
    value = layer_mean_distance(
        jnp.asarray(reference, dtype=jnp.float64),
        tuple(jnp.asarray(layer, dtype=jnp.float64) for layer in layers),
    )
    return float(jax.device_get(value))


def _job_path(stage: Path, dataset_name: str) -> Path:
    return stage / ".jobs" / f"{dataset_name}.csv"


def _checkpoint_epochs(manifest: Mapping[str, Any]) -> tuple[int, ...]:
    epochs = manifest.get("epochs")
    raw = manifest.get("checkpoint_epochs")
    if type(epochs) is not int or not isinstance(raw, list) or not raw:
        raise ValueError("invalid training checkpoint epochs")
    if any(type(epoch) is not int for epoch in raw):
        raise ValueError("invalid training checkpoint epochs")
    result = tuple(int(epoch) for epoch in raw)
    if any(epoch <= 0 or epoch > epochs for epoch in result):
        raise ValueError("invalid training checkpoint epochs")
    if result != tuple(sorted(set(result))):
        raise ValueError("invalid training checkpoint epochs")
    return result


def run_checkpoint_replay_job(
    config: AnalysisConfig,
    *,
    artifact_root: Path,
    dataset_name: str,
    seed: int,
) -> Path:
    if dataset_name not in config.datasets:
        raise ValueError(f"dataset is not enabled: {dataset_name}")
    root = Path(artifact_root)
    stage = feature_artifact_path(root, config.feature)
    stage.mkdir(parents=True, exist_ok=True)
    job_path = _job_path(stage, dataset_name)
    job_path.parent.mkdir(parents=True, exist_ok=True)
    if job_path.exists():
        raise FileExistsError(job_path)
    train_root = feature_artifact_path(root, "train_regularization_off")
    train_manifest = load_yaml(train_root / "manifest.yaml")
    checkpoint_epochs = _checkpoint_epochs(train_manifest)
    dataset = PreparedDataset(feature_artifact_path(root, "data_prep") / dataset_name)
    plan = build_model_plan(dataset)
    readouts = train_manifest.get("readout_by_dataset")
    if (
        not isinstance(readouts, Mapping)
        or readouts.get(dataset_name) != plan.readout.kind
    ):
        raise ValueError(f"training readout does not match replay plan: {dataset_name}")
    key = jax.random.key(int(seed))
    _, base_model_state = initialize(key, plan)
    kernel = _spectrum_kernel(plan)
    batch_size = int(config.batch_size[dataset_name])
    train_reference = _reference_spectrum(dataset_name, "train", root)
    test_reference = _reference_spectrum(dataset_name, "test", root)
    rows: list[dict[str, Any]] = []
    for checkpoint_index, epoch in enumerate(checkpoint_epochs, start=1):
        snapshot = restore_model_checkpoint(
            train_root / "checkpoints" / dataset_name / f"epoch_{epoch:03d}"
        )
        params = snapshot["params"]
        model_state = _restored_model_state(base_model_state, snapshot)
        train_layers = _layer_spectra(
            dataset,
            "train",
            batch_size=batch_size,
            kernel=kernel,
            params=params,
            model_state=model_state,
        )
        test_layers = _layer_spectra(
            dataset,
            "test",
            batch_size=batch_size,
            kernel=kernel,
            params=params,
            model_state=model_state,
        )
        rows.append(
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "epoch": epoch,
                "train_d_model": _distance(train_reference, train_layers),
                "test_d_model": _distance(test_reference, test_layers),
            }
        )
        print(
            f"{config.feature} dataset={dataset_name} "
            f"checkpoint={checkpoint_index}/{len(checkpoint_epochs)} epoch={epoch}",
            flush=True,
        )
    write_csv(job_path, D_MODEL_COLUMNS, rows)
    return job_path


def _read_job(
    path: Path,
    checkpoint_epochs: Sequence[int],
    *,
    dataset_name: str,
    seed: int,
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != D_MODEL_COLUMNS:
            raise ValueError(f"invalid D_model columns: {path}")
        rows = [dict(row) for row in reader]
    if len(rows) != len(checkpoint_epochs):
        raise ValueError(f"invalid D_model row count: {path}")
    distance_columns = D_MODEL_COLUMNS[3:]
    for expected_epoch, row in zip(checkpoint_epochs, rows, strict=True):
        try:
            row_seed = int(row["seed"])
            row_epoch = int(row["epoch"])
            distances = tuple(float(row[column]) for column in distance_columns)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid D_model value: {path}") from error
        if row["dataset"] != dataset_name:
            raise ValueError(f"invalid D_model dataset: {path}")
        if row_seed != int(seed):
            raise ValueError(f"invalid D_model seed: {path}")
        if row_epoch != expected_epoch:
            raise ValueError(f"invalid D_model epoch sequence: {path}")
        if not all(np.isfinite(value) and value >= 0.0 for value in distances):
            raise ValueError(f"invalid D_model distance: {path}")
    return rows


def finalize_checkpoint_replay(
    config: AnalysisConfig,
    *,
    artifact_root: Path,
    seed: int,
) -> Path:
    root = Path(artifact_root)
    stage = feature_artifact_path(root, config.feature)
    train_manifest = load_yaml(
        feature_artifact_path(root, "train_regularization_off") / "manifest.yaml"
    )
    checkpoint_epochs = _checkpoint_epochs(train_manifest)
    rows: list[dict[str, str]] = []
    for dataset_name in config.datasets:
        rows.extend(
            _read_job(
                _job_path(stage, dataset_name),
                checkpoint_epochs,
                dataset_name=dataset_name,
                seed=seed,
            )
        )
    output = stage / "d_model.csv"
    write_csv(output, D_MODEL_COLUMNS, rows)
    write_yaml(
        stage / "manifest.yaml",
        {
            "feature": config.feature,
            "datasets": list(config.datasets),
            "seed": int(seed),
            "epochs": int(train_manifest["epochs"]),
            "checkpoint_epochs": list(checkpoint_epochs),
            "batch_size": dict(config.batch_size),
            "representation": "dft_magnitude",
            "reference": "dataset_input",
            "signal_processing_dtype": "float64",
            "readout_by_dataset": {
                dataset: experiment_spec(dataset).readout for dataset in config.datasets
            },
        },
    )
    remove_tree(stage / ".jobs")
    return output


__all__ = [
    "D_MODEL_COLUMNS",
    "finalize_checkpoint_replay",
    "run_checkpoint_replay_job",
]
