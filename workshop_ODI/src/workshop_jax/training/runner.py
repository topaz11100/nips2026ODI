from __future__ import annotations

import csv
from pathlib import Path
import time
from typing import Any, Mapping

import jax
import numpy as np

from workshop_jax.artifacts import (
    feature_artifact_path,
    remove_tree,
    save_model_checkpoint,
    write_csv,
    write_yaml,
)
from workshop_jax.config import TrainConfig
from workshop_jax.data.dataset import (
    HostBatch,
    PreparedDataset,
    epoch_indices,
    index_batches,
)
from workshop_jax.models import initialize
from workshop_jax.paper_spec import build_model_plan, experiment_spec
from workshop_jax.training.steps import (
    DeviceBatch,
    MetricSums,
    TrainingState,
    build_steps,
    zero_sums,
)


TRAINING_COLUMNS = (
    "dataset",
    "seed",
    "lambda",
    "epoch",
    "train_loss",
    "train_accuracy",
    "test_loss",
    "test_accuracy",
)


def _device_batch(batch: HostBatch) -> DeviceBatch:
    return DeviceBatch(
        inputs=jax.device_put(batch.inputs),
        targets=jax.device_put(batch.targets),
        real_sample_mask=jax.device_put(batch.real_sample_mask),
    )


def _initial_state(plan: Any, steps: Any, seed: int) -> TrainingState:
    key = jax.random.key(int(seed))
    params, model_state = initialize(key, plan)
    return TrainingState(
        params=params,
        model_state=model_state,
        opt_state=steps.optimizer.init(params),
    )


def _metrics(value: MetricSums) -> tuple[float, float]:
    host = jax.device_get(value)
    count = int(host.count)
    if count <= 0:
        raise ValueError("metric accumulator contains no real samples")
    loss, accuracy = float(host.loss_sum) / count, float(host.correct_sum) / count
    if not np.isfinite(loss) or not np.isfinite(accuracy):
        raise FloatingPointError("nonfinite training or evaluation metrics")
    return loss, accuracy


def _evaluate(
    dataset: PreparedDataset,
    state: TrainingState,
    steps: Any,
    *,
    batch_size: int,
) -> tuple[float, float]:
    totals = zero_sums()
    indices = np.arange(dataset.split_count("test"), dtype=np.int64)
    for selected in index_batches(indices, batch_size):
        batch = dataset.read_batch("test", selected, batch_size=batch_size)
        totals = steps.eval_step(state, _device_batch(batch), totals)
    return _metrics(totals)


def _snapshot(state: TrainingState) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"params": state.params}
    if "normalization" in state.model_state:
        snapshot["model_state"] = {"normalization": state.model_state["normalization"]}
    return snapshot


def _job_path(stage: Path, dataset: str) -> Path:
    return stage / ".jobs" / f"{dataset}.csv"


def _checkpoint_epochs(config: TrainConfig) -> tuple[int, ...]:
    if config.feature != "train_regularization_off":
        return ()
    if config.checkpoint_epochs:
        return config.checkpoint_epochs
    return tuple(range(1, config.epochs + 1))


def run_training_job(
    config: TrainConfig,
    *,
    artifact_root: Path,
    dataset_name: str,
    seed: int,
) -> Path:
    if dataset_name not in config.datasets:
        raise ValueError(f"dataset is not enabled for {config.feature}: {dataset_name}")
    stage = feature_artifact_path(artifact_root, config.feature)
    stage.mkdir(parents=True, exist_ok=True)
    job_path = _job_path(stage, dataset_name)
    job_path.parent.mkdir(parents=True, exist_ok=True)
    if job_path.exists():
        raise FileExistsError(job_path)
    dataset = PreparedDataset(
        feature_artifact_path(artifact_root, "data_prep") / dataset_name
    )
    plan = build_model_plan(dataset)
    coefficient = float(config.regularization_lambda[dataset_name])
    steps = build_steps(
        plan,
        learning_rate=config.learning_rate,
        regularization_lambda=coefficient,
    )
    state = _initial_state(plan, steps, seed)
    batch_size = int(config.batch_size[dataset_name])
    train_count = dataset.split_count("train")
    usable_count = (train_count // batch_size) * batch_size
    if usable_count == 0:
        raise ValueError(f"training batch exceeds split size for {dataset_name}")
    checkpoints = stage / "checkpoints" / dataset_name
    checkpoint_epochs = frozenset(_checkpoint_epochs(config))
    if config.feature == "train_regularization_off":
        checkpoints.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    print(
        f"START {config.feature} dataset={dataset_name} seed={seed} "
        f"lambda={coefficient!r} batch_size={batch_size} epochs={config.epochs}",
        flush=True,
    )
    for epoch in range(1, config.epochs + 1):
        schedule = epoch_indices(train_count, seed=seed, epoch=epoch)[:usable_count]
        totals = zero_sums()
        for selected in index_batches(schedule, batch_size):
            batch = dataset.read_batch("train", selected, batch_size=batch_size)
            state, totals = steps.train_step(state, _device_batch(batch), totals)
        train_loss, train_accuracy = _metrics(totals)
        test_loss, test_accuracy = _evaluate(
            dataset,
            state,
            steps,
            batch_size=batch_size,
        )
        if epoch in checkpoint_epochs:
            save_model_checkpoint(
                checkpoints / f"epoch_{epoch:03d}",
                _snapshot(state),
            )
        rows.append(
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "lambda": coefficient,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )
        print(
            f"{config.feature} dataset={dataset_name} "
            f"epoch={epoch}/{config.epochs} "
            f"lambda={coefficient!r} "
            f"train_loss={train_loss:.8g} test_loss={test_loss:.8g} "
            f"train_accuracy={train_accuracy:.8g} test_accuracy={test_accuracy:.8g} "
            f"elapsed_seconds={time.monotonic() - started:.1f}",
            flush=True,
        )
    write_csv(job_path, TRAINING_COLUMNS, rows)
    return job_path


def _read_job(
    path: Path,
    expected_rows: int,
    *,
    dataset_name: str,
    seed: int,
    regularization_lambda: float,
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TRAINING_COLUMNS:
            raise ValueError(f"invalid training metric columns: {path}")
        rows = [dict(row) for row in reader]
    if len(rows) != expected_rows:
        raise ValueError(f"invalid training metric row count: {path}")
    for expected_epoch, row in enumerate(rows, start=1):
        try:
            row_seed = int(row["seed"])
            row_lambda = float(row["lambda"])
            row_epoch = int(row["epoch"])
            train_loss = float(row["train_loss"])
            train_accuracy = float(row["train_accuracy"])
            test_loss = float(row["test_loss"])
            test_accuracy = float(row["test_accuracy"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid training metric value: {path}") from error
        if row["dataset"] != dataset_name:
            raise ValueError(f"invalid training metric dataset: {path}")
        if row_seed != int(seed):
            raise ValueError(f"invalid training metric seed: {path}")
        if not np.isfinite(row_lambda) or row_lambda != float(regularization_lambda):
            raise ValueError(f"invalid training metric lambda: {path}")
        if row_epoch != expected_epoch:
            raise ValueError(f"invalid training metric epoch sequence: {path}")
        if not np.isfinite(train_loss) or train_loss < 0.0:
            raise ValueError(f"invalid training loss: {path}")
        if not np.isfinite(test_loss) or test_loss < 0.0:
            raise ValueError(f"invalid test loss: {path}")
        if not np.isfinite(train_accuracy) or not 0.0 <= train_accuracy <= 1.0:
            raise ValueError(f"invalid training accuracy: {path}")
        if not np.isfinite(test_accuracy) or not 0.0 <= test_accuracy <= 1.0:
            raise ValueError(f"invalid test accuracy: {path}")
    return rows


def finalize_training(
    config: TrainConfig,
    *,
    artifact_root: Path,
    seed: int,
) -> Path:
    stage = feature_artifact_path(artifact_root, config.feature)
    rows: list[Mapping[str, Any]] = []
    for dataset_name in config.datasets:
        rows.extend(
            _read_job(
                _job_path(stage, dataset_name),
                config.epochs,
                dataset_name=dataset_name,
                seed=seed,
                regularization_lambda=config.regularization_lambda[dataset_name],
            )
        )
        if config.feature == "train_regularization_off":
            checkpoint_root = stage / "checkpoints" / dataset_name
            expected = {f"epoch_{epoch:03d}" for epoch in _checkpoint_epochs(config)}
            actual = {path.name for path in checkpoint_root.iterdir() if path.is_dir()}
            if actual != expected:
                raise ValueError(f"invalid checkpoint set for {dataset_name}")
    if config.feature == "train_regularization_on" and (stage / "checkpoints").exists():
        raise ValueError("regularization-on training created checkpoints")
    output = stage / "training_metrics.csv"
    write_csv(output, TRAINING_COLUMNS, rows)
    manifest = {
        "feature": config.feature,
        "datasets": list(config.datasets),
        "seed": int(seed),
        "epochs": int(config.epochs),
        "learning_rate": float(config.learning_rate),
        "lambda": dict(config.regularization_lambda),
        "scenario_count": len(config.datasets),
        "batch_size": dict(config.batch_size),
        "reference": "dataset_input",
        "representation": "dft_magnitude",
        "temporal_centering": "sample_demean",
        "distance_metric": "centered_l2",
        "layer_aggregation": "arithmetic_mean",
        "parameter_dtype": "float32",
        "training_compute_dtype": "float32",
        "regularization_signal_dtype": "float32",
        "loss_accumulator_dtype": "float32",
        "metric_count_dtype": "int64",
        "surrogate_scale": 0.1,
        "readout_by_dataset": {
            dataset: experiment_spec(dataset).readout for dataset in config.datasets
        },
    }
    if config.feature == "train_regularization_off":
        manifest["checkpoint_epochs"] = list(_checkpoint_epochs(config))
    write_yaml(stage / "manifest.yaml", manifest)
    remove_tree(stage / ".jobs")
    return output


__all__ = [
    "TRAINING_COLUMNS",
    "finalize_training",
    "run_training_job",
]
