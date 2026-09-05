from __future__ import annotations

from pathlib import Path
import jax
import numpy as np

from workshop_jax.analysis.spectral import dft_magnitude_sum
from workshop_jax.artifacts import (
    atomic_directory,
    feature_artifact_path,
    validate_dataset_reference_file,
    write_npz,
    write_yaml,
)
from workshop_jax.config import AnalysisConfig
from workshop_jax.data.dataset import PreparedDataset, index_batches


def _split_spectrum(
    dataset: PreparedDataset,
    split: str,
    *,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    kernel = jax.jit(dft_magnitude_sum)
    total: np.ndarray | None = None
    component_count = np.float64(0.0)
    indices = dataset.probe_indices(split)
    for selected in index_batches(indices, batch_size):
        host = dataset.read_signal_batch(split, selected, batch_size=batch_size)
        current_total, current_count = jax.device_get(
            kernel(
                jax.device_put(host.inputs),
                jax.device_put(host.real_sample_mask),
            )
        )
        current = np.asarray(current_total, dtype=np.float64)
        if total is None:
            total = current.copy()
        else:
            np.add(total, current, out=total)
        component_count = np.float64(component_count + np.float64(current_count))
    if total is None or component_count <= 0.0:
        raise ValueError(f"empty probe for {dataset.dataset_id}/{split}")
    return total / component_count, int(indices.size)


def run_dataset_reference_job(
    config: AnalysisConfig,
    *,
    artifact_root: Path,
    dataset_name: str,
) -> Path:
    if dataset_name not in config.datasets:
        raise ValueError(f"dataset is not enabled: {dataset_name}")
    stage = feature_artifact_path(artifact_root, config.feature)
    stage.mkdir(parents=True, exist_ok=True)
    output = stage / dataset_name
    dataset = PreparedDataset(
        feature_artifact_path(artifact_root, "data_prep") / dataset_name
    )
    batch_size = int(config.batch_size[dataset_name])
    train_spectrum, train_count = _split_spectrum(
        dataset,
        "train",
        batch_size=batch_size,
    )
    test_spectrum, test_count = _split_spectrum(
        dataset,
        "test",
        batch_size=batch_size,
    )
    with atomic_directory(output) as temporary:
        write_npz(
            temporary / "reference_dft.npz",
            train_spectrum=train_spectrum,
            test_spectrum=test_spectrum,
            train_sample_count=np.asarray(train_count, dtype=np.int64),
            test_sample_count=np.asarray(test_count, dtype=np.int64),
            sequence_length=np.asarray(
                dataset.canonical_sample_shape[0], dtype=np.int64
            ),
        )
    return output / "reference_dft.npz"


def finalize_dataset_references(
    config: AnalysisConfig,
    *,
    artifact_root: Path,
) -> Path:
    stage = feature_artifact_path(artifact_root, config.feature)
    for dataset in config.datasets:
        reference_path = stage / dataset / "reference_dft.npz"
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        validate_dataset_reference_file(reference_path)
    manifest = stage / "manifest.yaml"
    write_yaml(
        manifest,
        {
            "feature": config.feature,
            "datasets": list(config.datasets),
            "batch_size": dict(config.batch_size),
            "representation": "dft_magnitude",
            "reference": "dataset_input",
            "signal_processing_dtype": "float64",
        },
    )
    return manifest


__all__ = [
    "finalize_dataset_references",
    "run_dataset_reference_job",
]
