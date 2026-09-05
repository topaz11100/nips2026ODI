from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping

import numpy as np

from workshop_jax.data.preparation.common import (
    canonical_source_root,
    create_dataset_writer,
    initialize_dataset_tree,
    write_dataset_metadata,
)
from workshop_jax.data.preparation.event_common import (
    ObservationInterval,
    accumulate_fixed_window_spatial_event_counts,
    event_preprocessing_metadata,
)
from workshop_jax.data.policy import (
    CIFAR10_DVS_PREPROCESSING_PROFILE,
    DatasetPolicy,
)


_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
_FIXED_INTERVAL_LOCATOR_ID = "cifar10dvs_aedat2_first_decoded_event_fixed_1p3s"


@dataclass(frozen=True)
class CIFAR10DVSSample:
    split: str
    label: int
    class_name: str
    source_ordinal: int
    source_path: Path
    sample_id: str


@dataclass(frozen=True)
class CIFAR10DVSSource:
    root: Path
    layout_id: str
    samples: Mapping[str, tuple[CIFAR10DVSSample, ...]]
    class_counts: tuple[int, ...]
    train_counts: tuple[int, ...]


def discover_cifar10_dvs_source(raw_root: Path) -> CIFAR10DVSSource:
    source = canonical_source_root(raw_root, "cifar10-dvs")
    for candidate in (source / "CIFAR10DVS", source):
        resolved = _try_class_tree(candidate)
        if resolved is not None:
            return resolved
    raise FileNotFoundError(
        "CIFAR10-DVS source must contain ten class directories of official "
        ".aedat files inside "
        "<raw_data>/cifar10-dvs/ or its CIFAR10DVS wrapper"
    )


def prepare_cifar10_dvs_dataset(
    raw_root: Path,
    policy: DatasetPolicy,
    dataset_root: Path,
) -> None:
    source = discover_cifar10_dvs_source(raw_root)
    counts = {split: len(source.samples[split]) for split in ("train", "test")}
    initialize_dataset_tree(dataset_root)
    split_class_counts: dict[str, tuple[int, ...]] = {}

    interval_locator_counts: dict[str, int] = {}
    decoded_event_count = 0
    for split in ("train", "test"):
        samples = source.samples[split]
        writer = create_dataset_writer(
            dataset_root,
            policy,
            split=split,
            sample_count=len(samples),
        )
        for index, sample in enumerate(samples):
            events, interval = _load_sample(sample)
            interval_locator_counts[interval.locator_id] = (
                interval_locator_counts.get(interval.locator_id, 0) + 1
            )
            decoded_event_count += int(events["t"].shape[0])
            frames = accumulate_fixed_window_spatial_event_counts(
                timestamps=events["t"],
                x=events["x"],
                y=events["y"],
                polarity=events["p"],
                interval=interval,
                num_steps=policy.num_steps,
                bin_width=CIFAR10_DVS_PREPROCESSING_PROFILE.bin_width_us,
                height=128,
                width=128,
            )
            writer.write_sample(
                index,
                frames,
                np.float64(sample.label),
                sample.sample_id,
            )
            del events, frames
        writer.commit()
        split_class_counts[split] = writer.class_counts

    write_dataset_metadata(
        dataset_root,
        policy,
        source={
            "adapter_id": policy.adapter_id,
            "source_root": str(source.root),
            "format": source.layout_id,
            "class_names": list(_CLASS_NAMES),
            "class_counts": list(source.class_counts),
            "class_train_counts": list(source.train_counts),
            "split_policy": "class_balanced_stable_source_order_90_10",
        },
        preprocessing=_cifar10_dvs_preprocessing_metadata(
            policy,
            decoded_event_count=decoded_event_count,
            interval_locator_counts=interval_locator_counts,
        ),
        split_counts=counts,
        split_class_counts=split_class_counts,
    )


def _cifar10_dvs_preprocessing_metadata(
    policy: DatasetPolicy,
    *,
    decoded_event_count: int,
    interval_locator_counts: Mapping[str, int],
) -> dict[str, object]:
    return event_preprocessing_metadata(
        policy_id=policy.policy_id,
        num_steps=policy.num_steps,
        reduction="event_count",
        interval_locator=_FIXED_INTERVAL_LOCATOR_ID,
        fixed_interval_seconds=(
            0.0,
            CIFAR10_DVS_PREPROCESSING_PROFILE.duration_s,
        ),
        spatial_shape=(128, 128),
        source_details={
            "split_rule": "class_balanced_stable_source_order_90_10",
            "decoded_event_count": decoded_event_count,
            "duration_us": CIFAR10_DVS_PREPROCESSING_PROFILE.duration_us,
            "bin_width_us": CIFAR10_DVS_PREPROCESSING_PROFILE.bin_width_us,
            "interval_locator_counts": dict(interval_locator_counts),
        },
    )


def _decode_cifar10_dvs_aedat_with_fixed_interval(
    path: Path,
) -> tuple[dict[str, np.ndarray], ObservationInterval]:
    with path.open("rb") as handle:
        _seek_aedat_payload(handle)
        payload = handle.read()
    records = np.frombuffer(
        payload,
        dtype=np.dtype([("address", ">u4"), ("timestamp", ">u4")]),
    )
    address = records["address"].astype(np.uint32, copy=False)
    timestamp = records["timestamp"].astype(np.int64)
    x = (127 - ((address & np.uint32(0x00FE)) >> np.uint32(1))).astype(np.int16)
    y = ((address & np.uint32(0x7F00)) >> np.uint32(8)).astype(np.int16)
    polarity = (1 - (address & np.uint32(1))).astype(np.int8)
    interval = _fixed_interval_from_timestamps(timestamp)
    return (
        {
            "t": timestamp,
            "x": x,
            "y": y,
            "p": polarity,
        },
        interval,
    )


def _try_class_tree(root: Path) -> CIFAR10DVSSource | None:
    if not root.is_dir():
        return None
    class_roots = _resolve_class_roots(root)
    if class_roots is None:
        return None
    per_class_paths: list[tuple[Path, ...]] = []
    for class_root in class_roots:
        aedat = tuple(
            sorted(path for path in class_root.rglob("*.aedat") if path.is_file())
        )
        per_class_paths.append(aedat)
    train: list[CIFAR10DVSSample] = []
    test: list[CIFAR10DVSSample] = []
    class_counts: list[int] = []
    train_counts: list[int] = []
    for label, paths in enumerate(per_class_paths):
        count = len(paths)
        train_count = int(np.ceil(0.9 * count))
        class_counts.append(count)
        train_counts.append(train_count)
        for ordinal, path in enumerate(paths):
            split = "train" if ordinal < train_count else "test"
            relative = path.relative_to(root).as_posix()
            sample = CIFAR10DVSSample(
                split=split,
                label=label,
                class_name=_CLASS_NAMES[label],
                source_ordinal=ordinal,
                source_path=path,
                sample_id=(
                    f"cifar10-dvs:{split}:class{label:02d}:"
                    f"source{ordinal:05d}:{relative}"
                ),
            )
            (train if split == "train" else test).append(sample)
    return CIFAR10DVSSource(
        root=root,
        layout_id="official_aedat_v2_class_tree",
        samples={"train": tuple(train), "test": tuple(test)},
        class_counts=tuple(class_counts),
        train_counts=tuple(train_counts),
    )


def _resolve_class_roots(root: Path) -> tuple[Path, ...] | None:
    named = tuple(root / name for name in _CLASS_NAMES)
    if all(path.is_dir() for path in named):
        return named
    numeric = tuple(root / str(index) for index in range(10))
    if all(path.is_dir() for path in numeric):
        return numeric
    return None


def _load_sample(
    sample: CIFAR10DVSSample,
) -> tuple[dict[str, np.ndarray], ObservationInterval]:
    return _decode_cifar10_dvs_aedat_with_fixed_interval(sample.source_path)


def _seek_aedat_payload(handle: BinaryIO) -> int:
    while True:
        position = handle.tell()
        marker = handle.read(1)
        if marker == b"#":
            handle.readline()
            continue
        handle.seek(position)
        return int(position)


def _fixed_interval_from_timestamps(
    timestamp: np.ndarray,
) -> ObservationInterval:
    values = np.asarray(timestamp).reshape(-1)
    start = int(values[0])
    return ObservationInterval(
        start=start,
        end=start + CIFAR10_DVS_PREPROCESSING_PROFILE.duration_us,
        locator_id=_FIXED_INTERVAL_LOCATOR_ID,
    )


__all__ = [
    "CIFAR10DVSSample",
    "CIFAR10DVSSource",
    "prepare_cifar10_dvs_dataset",
]
