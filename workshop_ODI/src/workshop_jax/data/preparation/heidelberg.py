from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from workshop_jax.data.policy import DatasetPolicy, SHD_PREPROCESSING_PROFILE
from workshop_jax.data.preparation.common import (
    canonical_source_root,
    create_dataset_writer,
    initialize_dataset_tree,
    write_dataset_metadata,
)
from workshop_jax.data.preparation.event_common import (
    ObservationInterval,
    accumulate_fixed_window_pooled_channel_counts,
    event_preprocessing_metadata,
)


SHD_INTERVAL_LOCATOR_ID = "fixed_absolute_recording_origin_zero_to_one_second"


def _find_file(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    expected = [str(root / name) for name in names]
    raise FileNotFoundError(f"expected one of {expected}")


def _source_files(raw_root: Path) -> tuple[Path, Path, Path]:
    dataset_root = canonical_source_root(raw_root, "shd")
    candidates = (dataset_root / "SHD", dataset_root)
    source = dataset_root
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("shd_train.h5*")):
            source = candidate.resolve(strict=False)
            break
    train = _find_file(source, ("shd_train.h5", "shd_train.hdf5"))
    test = _find_file(source, ("shd_test.h5", "shd_test.hdf5"))
    return source, train, test


def _open_h5(path: Path) -> Any:
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("SHD preprocessing requires h5py") from error
    return h5py.File(path, "r")


def _datasets(handle: Any) -> tuple[Any, Any, Any]:
    try:
        return handle["spikes/times"], handle["spikes/units"], handle["labels"]
    except KeyError as error:
        raise ValueError(
            "SHD HDF5 must contain spikes/times, spikes/units, and labels"
        ) from error


def _preprocessing_metadata(interval_counts: dict[str, int]) -> dict[str, Any]:
    profile = SHD_PREPROCESSING_PROFILE
    return event_preprocessing_metadata(
        policy_id=profile.preprocessing_version,
        num_steps=profile.time_steps,
        reduction="event_count",
        interval_locator=SHD_INTERVAL_LOCATOR_ID,
        fixed_interval_seconds=(0.0, profile.duration_s),
        spatial_shape=None,
        source_details={
            "interval_locator_counts": interval_counts,
            "raw_channels": profile.raw_channels,
            "channels": profile.input_channels,
            "channel_pool_size": profile.channel_pool_size,
            "duration_s": profile.duration_s,
            "bin_width_s": profile.time_bin_s,
        },
    )


def prepare_heidelberg_dataset(
    raw_root: Path,
    policy: DatasetPolicy,
    dataset_root: Path,
) -> None:
    if policy.dataset_id != "shd":
        raise ValueError(f"unsupported Heidelberg dataset: {policy.dataset_id}")
    source, train_path, test_path = _source_files(raw_root)
    paths = {"train": train_path, "test": test_path}
    counts: dict[str, int] = {}
    for split, path in paths.items():
        with _open_h5(path) as handle:
            _, _, labels = _datasets(handle)
            counts[split] = int(labels.shape[0])
    initialize_dataset_tree(dataset_root)
    class_counts: dict[str, tuple[int, ...]] = {}
    interval = ObservationInterval(
        start=0.0,
        end=SHD_PREPROCESSING_PROFILE.duration_s,
        locator_id=SHD_INTERVAL_LOCATOR_ID,
    )
    interval_counts = {SHD_INTERVAL_LOCATOR_ID: sum(counts.values())}
    raw_channels = SHD_PREPROCESSING_PROFILE.raw_channels
    if raw_channels is None:
        raise ValueError("SHD preprocessing requires a raw channel count")
    for split, path in paths.items():
        writer = create_dataset_writer(
            dataset_root,
            policy,
            split=split,
            sample_count=counts[split],
        )
        with _open_h5(path) as handle:
            times, units, labels = _datasets(handle)
            for index in range(counts[split]):
                timestamps = np.asarray(times[index], dtype=np.float64).reshape(-1)
                channels = np.asarray(units[index], dtype=np.int64).reshape(-1)
                frames = accumulate_fixed_window_pooled_channel_counts(
                    timestamps_seconds=timestamps,
                    channels=channels,
                    interval=interval,
                    num_steps=SHD_PREPROCESSING_PROFILE.time_steps,
                    bin_width_seconds=SHD_PREPROCESSING_PROFILE.time_bin_s,
                    raw_num_channels=raw_channels,
                    channel_pool_size=SHD_PREPROCESSING_PROFILE.channel_pool_size,
                )
                writer.write_sample(
                    index,
                    frames[:, np.newaxis, np.newaxis, :],
                    int(np.asarray(labels[index]).reshape(())),
                    "",
                )
        writer.commit()
        class_counts[split] = writer.class_counts
    write_dataset_metadata(
        dataset_root,
        policy,
        source={
            "adapter_id": policy.adapter_id,
            "source_root": str(source),
            "format": "official_shd_hdf5",
            "train_file": train_path.name,
            "test_file": test_path.name,
            "source_split_counts": {
                "train": counts["train"],
                "test": counts["test"],
            },
        },
        preprocessing=_preprocessing_metadata(interval_counts),
        split_counts=counts,
        split_class_counts=class_counts,
    )


__all__ = ["prepare_heidelberg_dataset"]
