from __future__ import annotations

from pathlib import Path
from typing import Callable

from workshop_jax.artifacts import atomic_directory, feature_artifact_path, write_yaml
from workshop_jax.config import DataPrepConfig
from workshop_jax.data.policy import DatasetPolicy, dataset_policy
from workshop_jax.data.preparation.cifar10_dvs import prepare_cifar10_dvs_dataset
from workshop_jax.data.preparation.dvs128_gesture import prepare_dvs128_gesture_dataset
from workshop_jax.data.preparation.heidelberg import prepare_heidelberg_dataset
from workshop_jax.data.preparation.static_images import (
    prepare_cifar_dataset,
    prepare_mnist_dataset,
)
from workshop_jax.data.probes import materialize_probes


Preparer = Callable[[Path, DatasetPolicy, Path], None]


def _preparer(policy: DatasetPolicy) -> Preparer:
    return {
        "mnist": prepare_mnist_dataset,
        "cifar": prepare_cifar_dataset,
        "shd": prepare_heidelberg_dataset,
        "cifar10-dvs": prepare_cifar10_dvs_dataset,
        "dvs128-gesture": prepare_dvs128_gesture_dataset,
    }[policy.adapter_id]


def run_data_prep(
    config: DataPrepConfig,
    *,
    project_root: Path,
    artifact_root: Path,
    seed: int,
) -> Path:
    raw_root = Path(project_root) / "data" / "raw_data"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw data directory does not exist: {raw_root}")
    output = feature_artifact_path(artifact_root, config.feature)
    with atomic_directory(output) as temporary:
        for dataset_name in config.datasets:
            policy = dataset_policy(dataset_name)
            dataset_root = temporary / dataset_name
            dataset_root.mkdir()
            _preparer(policy)(raw_root, policy, dataset_root)
            materialize_probes(
                dataset_root,
                num_classes=policy.num_classes,
                train_size=int(config.train_probe_size[dataset_name]),
                test_size=int(config.test_probe_size[dataset_name]),
                seed=seed,
            )
        write_yaml(
            temporary / "manifest.yaml",
            {
                "feature": config.feature,
                "datasets": list(config.datasets),
                "seed": int(seed),
                "train_probe_size": dict(config.train_probe_size),
                "test_probe_size": dict(config.test_probe_size),
                "preprocessing_compute_dtype": "float64",
                "storage_dtype": "float64",
            },
        )
    return output


__all__ = ["run_data_prep"]
