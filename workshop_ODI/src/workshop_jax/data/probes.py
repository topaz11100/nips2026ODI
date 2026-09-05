from __future__ import annotations

from pathlib import Path

import numpy as np

from workshop_jax.artifacts import write_npy


def _selection_seed(seed: int, split: str) -> np.random.SeedSequence:
    split_code = {"train": 1, "test": 2}[split]
    return np.random.SeedSequence(int(seed), spawn_key=(split_code, 2, 0))


def _probe_indices(
    labels: np.ndarray,
    *,
    num_classes: int,
    total: int,
    seed: np.random.SeedSequence,
) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if total > values.size:
        raise ValueError(f"probe size {total} exceeds split size {values.size}")
    by_class = tuple(
        np.flatnonzero(values == class_id).astype(np.int64, copy=False)
        for class_id in range(num_classes)
    )
    counts = np.asarray([indices.size for indices in by_class], dtype=np.int64)
    targets = total * counts.astype(np.float64) / np.float64(values.size)
    quotas = np.floor(targets).astype(np.int64)
    remaining = int(total - np.sum(quotas))
    remainders = targets - quotas.astype(np.float64)
    order = np.lexsort((np.arange(num_classes, dtype=np.int64), -remainders))
    quotas[order[:remaining]] += 1
    selected: list[np.ndarray] = []
    for class_id, (eligible, quota) in enumerate(zip(by_class, quotas, strict=True)):
        if int(quota) > eligible.size:
            raise ValueError(f"probe quota exceeds class {class_id} population")
        class_seed = np.random.SeedSequence(
            seed.entropy,
            spawn_key=(*seed.spawn_key, class_id),
        )
        permutation = np.random.default_rng(class_seed).permutation(eligible.size)
        selected.append(eligible[permutation[: int(quota)]])
    return np.concatenate(selected).astype(np.int64, copy=False)


def materialize_probes(
    dataset_root: Path,
    *,
    num_classes: int,
    train_size: int,
    test_size: int,
    seed: int,
) -> None:
    root = Path(dataset_root)
    for split, total in (("train", train_size), ("test", test_size)):
        labels = np.load(root / split / "y.npy", mmap_mode="r", allow_pickle=False)
        indices = _probe_indices(
            labels,
            num_classes=num_classes,
            total=int(total),
            seed=_selection_seed(seed, split),
        )
        write_npy(root / f"{split}_probe_indices.npy", indices)


def load_probe_indices(dataset_root: Path, split: str) -> np.ndarray:
    return np.load(
        Path(dataset_root) / f"{split}_probe_indices.npy",
        mmap_mode="r",
        allow_pickle=False,
    )


__all__ = ["load_probe_indices", "materialize_probes"]
