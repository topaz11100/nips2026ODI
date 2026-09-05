from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from workshop_jax.artifacts import load_yaml
from workshop_jax.data.probes import load_probe_indices


@dataclass(frozen=True)
class HostBatch:
    inputs: np.ndarray
    targets: np.ndarray
    real_sample_mask: np.ndarray


class PreparedDataset:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        self.metadata = load_yaml(self.root / "dataset_metadata.yaml")
        self._inputs: dict[str, np.memmap] = {}
        self._targets: dict[str, np.memmap] = {}

    @property
    def dataset_id(self) -> str:
        return str(self.metadata["dataset"])

    @property
    def canonical_sample_shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.metadata["sample_shape"])

    @property
    def num_classes(self) -> int:
        return int(self.metadata["num_classes"])

    def split_count(self, split: str) -> int:
        return int(self.metadata["split_counts"][split])

    def _open_inputs(self, split: str) -> np.memmap:
        if split not in self._inputs:
            self._inputs[split] = np.load(
                self.root / split / "x.npy", mmap_mode="r", allow_pickle=False
            )
        return self._inputs[split]

    def _open_targets(self, split: str) -> np.memmap:
        if split not in self._targets:
            self._targets[split] = np.load(
                self.root / split / "y.npy", mmap_mode="r", allow_pickle=False
            )
        return self._targets[split]

    def read_batch(
        self,
        split: str,
        indices: np.ndarray,
        *,
        batch_size: int,
    ) -> HostBatch:
        return self._read_batch(
            split,
            indices,
            batch_size=batch_size,
            input_dtype=np.float32,
        )

    def read_signal_batch(
        self,
        split: str,
        indices: np.ndarray,
        *,
        batch_size: int,
    ) -> HostBatch:
        return self._read_batch(
            split,
            indices,
            batch_size=batch_size,
            input_dtype=np.float64,
        )

    def _read_batch(
        self,
        split: str,
        indices: np.ndarray,
        *,
        batch_size: int,
        input_dtype: Any,
    ) -> HostBatch:
        selected = np.asarray(indices, dtype=np.int64)
        real_count = int(selected.size)
        if real_count > batch_size:
            raise ValueError("batch contains more indices than the configured size")
        x_source = self._open_inputs(split)
        y_source = self._open_targets(split)
        inputs = np.zeros((batch_size, *x_source.shape[1:]), dtype=input_dtype)
        targets = np.zeros((batch_size,), dtype=np.int32)
        mask = np.zeros((batch_size,), dtype=np.bool_)
        if real_count:
            inputs[:real_count] = np.asarray(x_source[selected], dtype=input_dtype)
            targets[:real_count] = np.rint(
                np.asarray(y_source[selected], dtype=np.float64)
            ).astype(np.int32)
            mask[:real_count] = True
        return HostBatch(inputs, targets, mask)

    def probe_indices(self, split: str) -> np.ndarray:
        return load_probe_indices(self.root, split)


def index_batches(indices: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    values = np.asarray(indices, dtype=np.int64)
    for start in range(0, values.size, batch_size):
        yield values[start : min(start + batch_size, values.size)]


def epoch_indices(sample_count: int, *, seed: int, epoch: int) -> np.ndarray:
    sequence = np.random.SeedSequence([seed, epoch, 0x524A4158])
    generator = np.random.Generator(np.random.PCG64(sequence))
    return generator.permutation(sample_count).astype(np.int64, copy=False)


__all__ = [
    "HostBatch",
    "PreparedDataset",
    "epoch_indices",
    "index_batches",
]
