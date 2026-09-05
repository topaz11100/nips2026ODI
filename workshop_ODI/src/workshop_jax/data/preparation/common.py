from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from workshop_jax.artifacts import create_memmap, write_yaml
from workshop_jax.data.policy import DatasetPolicy


DIRTY_FLUSH_BYTES = 256 * 1024 * 1024


def canonical_source_root(raw_root: Path, source_dataset_id: str) -> Path:
    return Path(raw_root).expanduser() / str(source_dataset_id)


def initialize_dataset_tree(dataset_root: Path) -> None:
    (dataset_root / "train").mkdir(parents=True, exist_ok=False)
    (dataset_root / "test").mkdir(parents=True, exist_ok=False)


def _portable_source(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in source.items()
        if str(key) not in {"source_root", "train_file", "test_file"}
    }


def write_dataset_metadata(
    dataset_root: Path,
    policy: DatasetPolicy,
    *,
    source: dict[str, Any],
    preprocessing: dict[str, Any],
    split_counts: dict[str, int],
    split_class_counts: Mapping[str, Sequence[int]],
) -> None:
    metadata = {
        "dataset": policy.selector,
        "preprocessing_policy": policy.policy_id,
        "storage_dtype": "float64",
        "sample_shape": [int(value) for value in policy.sample_shape],
        "num_classes": int(policy.num_classes),
        "split_counts": {split: int(count) for split, count in split_counts.items()},
        "split_class_counts": {
            split: [int(value) for value in counts]
            for split, counts in split_class_counts.items()
        },
    }
    preprocessing_payload = {
        "dataset": policy.selector,
        "policy": policy.policy_id,
        "source": _portable_source(source),
        "transform": dict(preprocessing),
    }
    write_yaml(dataset_root / "dataset_metadata.yaml", metadata)
    write_yaml(dataset_root / "preprocessing.yaml", preprocessing_payload)


class DatasetSplitWriter:
    def __init__(
        self,
        dataset_root: Path,
        policy: DatasetPolicy,
        *,
        split: str,
        sample_count: int,
    ) -> None:
        self.policy = policy
        self.split = split
        self.sample_count = int(sample_count)
        split_root = dataset_root / split
        self.x = create_memmap(
            split_root / "x.npy",
            shape=(self.sample_count, *policy.sample_shape),
            dtype=np.float64,
            preallocate=True,
        )
        self.y = create_memmap(
            split_root / "y.npy",
            shape=(self.sample_count,),
            dtype=np.float64,
            preallocate=True,
        )
        self._next_index = 0
        self._dirty_bytes = 0
        self._class_counts = np.zeros(policy.num_classes, dtype=np.int64)

    @property
    def class_counts(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self._class_counts)

    def write_sample(
        self,
        index: int,
        canonical: np.ndarray,
        target: Any,
        sample_id: str,
    ) -> None:
        del sample_id
        if int(index) != self._next_index:
            raise ValueError(
                f"nonsequential sample index for {self.policy.selector}/{self.split}"
            )
        sample = np.asarray(canonical, dtype=np.float64)
        if sample.shape != self.policy.sample_shape:
            raise ValueError(
                f"invalid sample shape for {self.policy.selector}: {sample.shape}"
            )
        target_array = np.asarray(target, dtype=np.float64)
        storage_bytes = int(sample.nbytes + target_array.nbytes)
        if self._dirty_bytes and self._dirty_bytes + storage_bytes > DIRTY_FLUSH_BYTES:
            self.flush()
        np.copyto(self.x[index], sample, casting="no")
        self.y[index] = target_array
        label = int(target_array.reshape(-1)[0])
        self._class_counts[label] += 1
        self._next_index += 1
        self._dirty_bytes += storage_bytes

    def flush(self) -> None:
        self.x.flush()
        self.y.flush()
        self._dirty_bytes = 0

    def commit(self) -> None:
        if self._next_index != self.sample_count:
            raise ValueError(
                f"incomplete split {self.policy.selector}/{self.split}: "
                f"{self._next_index}/{self.sample_count}"
            )
        self.flush()
        for array in (self.x, self.y):
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()


def create_dataset_writer(
    dataset_root: Path,
    policy: DatasetPolicy,
    *,
    split: str,
    sample_count: int,
) -> DatasetSplitWriter:
    return DatasetSplitWriter(
        dataset_root,
        policy,
        split=split,
        sample_count=sample_count,
    )


__all__ = [
    "DatasetSplitWriter",
    "canonical_source_root",
    "create_dataset_writer",
    "initialize_dataset_tree",
    "write_dataset_metadata",
]
