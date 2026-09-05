from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import yaml


REFERENCE_FILE_KEYS = {
    "train_spectrum",
    "test_spectrum",
    "train_sample_count",
    "test_sample_count",
    "sequence_length",
}


def load_yaml(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml_bytes(value: Any) -> bytes:
    return yaml.safe_dump(
        value,
        allow_unicode=True,
        sort_keys=False,
    ).encode("utf-8")


def _temporary_file(path: Path) -> tuple[int, Path]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        dir=target.parent,
    )
    return descriptor, Path(name)


def write_file(path: Path, payload: bytes) -> None:
    descriptor, temporary = _temporary_file(Path(path))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_yaml(path: Path, value: Any) -> None:
    write_file(path, dump_yaml_bytes(value))


def write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    descriptor, temporary = _temporary_file(Path(path))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(columns), extrasaction="raise"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def create_memmap(
    path: Path,
    *,
    shape: tuple[int, ...],
    dtype: Any,
    preallocate: bool = True,
) -> np.memmap:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        target,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )
    if preallocate:
        output.flush()
        with target.open("r+b") as handle:
            os.posix_fallocate(handle.fileno(), 0, target.stat().st_size)
    return output


def write_npy(path: Path, value: Any) -> None:
    descriptor, temporary = _temporary_file(Path(path))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.asarray(value), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_npz(path: Path, **arrays: Any) -> None:
    descriptor, temporary = _temporary_file(Path(path))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            savez: Any = np.savez
            savez(handle, **{name: np.asarray(value) for name, value in arrays.items()})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def atomic_directory(path: Path) -> Iterator[Path]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        yield temporary
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def initialize_run_root(
    root: Path,
    *,
    seed: int,
    features: Sequence[str],
) -> None:
    target = Path(root)
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(target)
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise FileExistsError(f"artifact root is not empty: {target}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    feature_directories = {
        str(feature): f"{feature}_{timestamp}" for feature in features
    }
    write_yaml(
        target / "run_manifest.yaml",
        {
            "seed": int(seed),
            "timestamp_utc": timestamp,
            "feature_directories": feature_directories,
        },
    )


def feature_artifact_path(root: Path, feature: str) -> Path:
    artifact_root = Path(root).expanduser().resolve(strict=True)
    manifest = load_yaml(artifact_root / "run_manifest.yaml")
    if not isinstance(manifest, Mapping):
        raise ValueError("run manifest must be a mapping")
    directories = manifest.get("feature_directories")
    if not isinstance(directories, Mapping) or feature not in directories:
        raise ValueError(f"run manifest has no feature directory for {feature}")
    directory = directories[feature]
    if not isinstance(directory, str) or Path(directory).name != directory:
        raise ValueError(f"invalid feature directory for {feature}")
    if not directory.startswith(f"{feature}_"):
        raise ValueError(f"invalid feature directory for {feature}")
    return artifact_root / directory


def reusable_run_root(
    root: Path,
    *,
    required_feature: str,
    expected_datasets: Sequence[str],
) -> tuple[Path, int]:
    target = Path(root).expanduser().resolve(strict=True)
    if not target.is_dir():
        raise NotADirectoryError(target)
    manifest = load_yaml(target / "run_manifest.yaml")
    if not isinstance(manifest, Mapping) or type(manifest.get("seed")) is not int:
        raise ValueError("invalid reusable run manifest")
    seed = int(manifest["seed"])
    feature_root = feature_artifact_path(target, required_feature)
    feature_manifest_path = feature_root / "manifest.yaml"
    if not feature_manifest_path.is_file():
        raise FileNotFoundError(feature_manifest_path)
    feature_manifest = load_yaml(feature_manifest_path)
    if not isinstance(feature_manifest, Mapping):
        raise ValueError(f"invalid {required_feature} manifest")
    if feature_manifest.get("feature") != required_feature:
        raise ValueError(f"invalid {required_feature} feature identity")
    if feature_manifest.get("seed") != seed:
        raise ValueError(f"invalid {required_feature} seed")
    if feature_manifest.get("datasets") != list(expected_datasets):
        raise ValueError(f"invalid {required_feature} dataset inventory")
    return target, seed


def validate_prepared_data_stage(
    root: Path,
    *,
    datasets: Sequence[str],
    seed: int,
    train_probe_size: Mapping[str, int],
    test_probe_size: Mapping[str, int],
) -> None:
    stage = feature_artifact_path(root, "data_prep")
    manifest = load_yaml(stage / "manifest.yaml")
    if not isinstance(manifest, Mapping):
        raise ValueError("invalid data_prep manifest")
    expected_manifest = {
        "feature": "data_prep",
        "datasets": list(datasets),
        "seed": int(seed),
        "train_probe_size": dict(train_probe_size),
        "test_probe_size": dict(test_probe_size),
        "preprocessing_compute_dtype": "float64",
        "storage_dtype": "float64",
    }
    if dict(manifest) != expected_manifest:
        raise ValueError("data_prep manifest does not match the current paper config")
    for dataset_name in datasets:
        dataset_root = stage / dataset_name
        metadata = load_yaml(dataset_root / "dataset_metadata.yaml")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"invalid prepared dataset metadata: {dataset_name}")
        if metadata.get("dataset") != dataset_name:
            raise ValueError(f"invalid prepared dataset identity: {dataset_name}")
        if metadata.get("storage_dtype") != "float64":
            raise ValueError(f"invalid prepared storage dtype: {dataset_name}")
        if not (dataset_root / "preprocessing.yaml").is_file():
            raise FileNotFoundError(dataset_root / "preprocessing.yaml")
        sample_shape_value = metadata.get("sample_shape")
        split_counts_value = metadata.get("split_counts")
        if not isinstance(sample_shape_value, list) or not isinstance(
            split_counts_value, Mapping
        ):
            raise ValueError(f"invalid prepared dataset shape: {dataset_name}")
        sample_shape = tuple(int(value) for value in sample_shape_value)
        if not sample_shape or any(value <= 0 for value in sample_shape):
            raise ValueError(f"invalid prepared dataset shape: {dataset_name}")
        for split, probe_sizes in (
            ("train", train_probe_size),
            ("test", test_probe_size),
        ):
            split_count = int(split_counts_value[split])
            inputs = np.load(
                dataset_root / split / "x.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            targets = np.load(
                dataset_root / split / "y.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            if inputs.shape != (split_count, *sample_shape):
                raise ValueError(
                    f"invalid prepared input shape: {dataset_name}/{split}"
                )
            if inputs.dtype != np.dtype(np.float64):
                raise ValueError(
                    f"invalid prepared input dtype: {dataset_name}/{split}"
                )
            if targets.shape != (split_count,):
                raise ValueError(
                    f"invalid prepared target shape: {dataset_name}/{split}"
                )
            if targets.dtype != np.dtype(np.float64):
                raise ValueError(
                    f"invalid prepared target dtype: {dataset_name}/{split}"
                )
            probe_path = dataset_root / f"{split}_probe_indices.npy"
            probe = np.load(probe_path, mmap_mode="r", allow_pickle=False)
            expected_size = int(probe_sizes[dataset_name])
            if probe.shape != (expected_size,) or probe.dtype != np.dtype(np.int64):
                raise ValueError(f"invalid prepared probe: {dataset_name}/{split}")
            values = np.asarray(probe)
            if (
                np.any(values < 0)
                or np.any(values >= split_count)
                or np.unique(values).size != expected_size
            ):
                raise ValueError(f"invalid prepared probe: {dataset_name}/{split}")


def validate_dataset_reference_file(path: Path) -> None:
    try:
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != REFERENCE_FILE_KEYS:
                raise ValueError(f"invalid dataset reference keys: {path}")
            sequence_length = np.asarray(stored["sequence_length"])
            if (
                sequence_length.shape != ()
                or sequence_length.dtype != np.dtype(np.int64)
                or int(sequence_length) <= 1
            ):
                raise ValueError(f"invalid dataset reference length: {path}")
            expected_shape = (int(sequence_length) // 2 + 1,)
            for split in ("train", "test"):
                sample_count = np.asarray(stored[f"{split}_sample_count"])
                if (
                    sample_count.shape != ()
                    or sample_count.dtype != np.dtype(np.int64)
                    or int(sample_count) <= 0
                ):
                    raise ValueError(f"invalid dataset reference count: {path}")
                values = np.asarray(stored[f"{split}_spectrum"])
                if values.shape != expected_shape or values.dtype != np.dtype(
                    np.float64
                ):
                    raise ValueError(f"invalid dataset reference curve: {path}")
                if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                    raise ValueError(f"invalid dataset reference curve: {path}")
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            "invalid dataset reference"
        ):
            raise
        raise ValueError(f"invalid dataset reference file: {path}") from error


def validate_dataset_signal_stage(
    root: Path,
    *,
    datasets: Sequence[str],
    batch_size: Mapping[str, int],
) -> None:
    stage = feature_artifact_path(root, "dataset_signal_analysis")
    manifest = load_yaml(stage / "manifest.yaml")
    if not isinstance(manifest, Mapping):
        raise ValueError("invalid dataset_signal_analysis manifest")
    expected_manifest = {
        "feature": "dataset_signal_analysis",
        "datasets": list(datasets),
        "batch_size": dict(batch_size),
        "representation": "dft_magnitude",
        "reference": "dataset_input",
        "signal_processing_dtype": "float64",
    }
    if dict(manifest) != expected_manifest:
        raise ValueError(
            "dataset_signal_analysis manifest does not match the current paper config"
        )
    for dataset_name in datasets:
        validate_dataset_reference_file(stage / dataset_name / "reference_dft.npz")


def archive_feature_stage(root: Path, feature: str) -> Path | None:
    target = feature_artifact_path(root, feature)
    if not target.exists() and not target.is_symlink():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived = target.with_name(f"{target.name}.previous_{timestamp}")
    if archived.exists() or archived.is_symlink():
        raise FileExistsError(archived)
    os.replace(target, archived)
    return archived


def save_model_checkpoint(path: Path, snapshot: Mapping[str, Any]) -> None:
    import orbax.checkpoint as ocp

    with atomic_directory(path) as temporary:
        payload = temporary / "model"
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(payload, dict(snapshot), force=False)
            checkpointer.wait_until_finished()


def restore_model_checkpoint(path: Path) -> Mapping[str, Any]:
    import orbax.checkpoint as ocp

    with ocp.StandardCheckpointer() as checkpointer:
        value = checkpointer.restore(Path(path) / "model")
    if not isinstance(value, Mapping):
        raise TypeError(f"invalid model checkpoint: {path}")
    return value


def remove_tree(path: Path) -> None:
    target = Path(path)
    if target.exists():
        shutil.rmtree(target)


__all__ = [
    "atomic_directory",
    "archive_feature_stage",
    "create_memmap",
    "dump_yaml_bytes",
    "feature_artifact_path",
    "initialize_run_root",
    "load_yaml",
    "remove_tree",
    "reusable_run_root",
    "restore_model_checkpoint",
    "save_model_checkpoint",
    "validate_dataset_reference_file",
    "validate_dataset_signal_stage",
    "validate_prepared_data_stage",
    "write_csv",
    "write_file",
    "write_npy",
    "write_npz",
    "write_yaml",
]
