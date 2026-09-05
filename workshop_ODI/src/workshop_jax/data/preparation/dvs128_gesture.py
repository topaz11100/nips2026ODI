from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import struct
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
    event_preprocessing_metadata,
    fixed_width_duration_indices,
)
from workshop_jax.data.policy import (
    DVS128_GESTURE_PREPROCESSING_PROFILE,
    DatasetPolicy,
)


@dataclass(frozen=True)
class GestureSample:
    split: str
    label: int
    sample_id: str
    source_path: Path
    interval_start: int | None = None
    interval_end: int | None = None
    trial_path: Path | None = None
    row_index: int | None = None


@dataclass(frozen=True)
class GestureSource:
    root: Path
    layout_id: str
    samples: Mapping[str, tuple[GestureSample, ...]]


@dataclass(frozen=True)
class _PolarityPacketIndex:
    data_offset: int
    event_size: int
    event_number: int
    ts_overflow: int
    min_timestamp: int | None
    max_timestamp: int | None


_AEDAT_EVENT_BLOCK_BYTES = 4 << 20
_FIXED_INTERVAL_LOCATOR_ID = "official_annotation_start_fixed_first_6s_usec"
_SELECTION_INTERVAL_LOCATOR_ID = "official_label_csv_start_end_usec"


def discover_dvs128_gesture_source(raw_root: Path) -> GestureSource:
    source = canonical_source_root(raw_root, "dvs128-gesture")
    aedat = _try_aedat_layout(source)
    if aedat is not None:
        return aedat
    raise FileNotFoundError(
        "DVS128 Gesture source under <raw_data>/dvs128-gesture/ must be an "
        "official DvsGesture directory with trials_to_train.txt, "
        "trials_to_test.txt, AEDAT files, and *_labels.csv"
    )


def prepare_dvs128_gesture_dataset(
    raw_root: Path,
    policy: DatasetPolicy,
    dataset_root: Path,
) -> None:
    source = discover_dvs128_gesture_source(raw_root)
    counts = {split: len(source.samples[split]) for split in ("train", "test")}
    initialize_dataset_tree(dataset_root)
    split_class_counts: dict[str, tuple[int, ...]] = {}

    locator_counts: dict[str, int] = {}
    for split in ("train", "test"):
        samples = source.samples[split]
        writer = create_dataset_writer(
            dataset_root,
            policy,
            split=split,
            sample_count=len(samples),
        )
        current_trial: Path | None = None
        current_packet_index: tuple[_PolarityPacketIndex, ...] = ()
        for index, sample in enumerate(samples):
            trial_path = sample.trial_path
            if trial_path is None:
                raise ValueError(
                    f"gesture sample has no trial path: {sample.sample_id}"
                )
            if trial_path != current_trial:
                current_packet_index = _index_aedat_v3_polarity_packets(trial_path)
                current_trial = trial_path
            support_interval, selection_interval = _sample_intervals(sample)
            locator_counts[selection_interval.locator_id] = (
                locator_counts.get(selection_interval.locator_id, 0) + 1
            )
            frames = _accumulate_gesture_segment(
                trial_path,
                packet_index=current_packet_index,
                support_interval=support_interval,
                selection_interval=selection_interval,
                num_steps=policy.num_steps,
            )
            writer.write_sample(
                index,
                frames,
                np.float64(sample.label),
                sample.sample_id,
            )
            del frames
        writer.commit()
        split_class_counts[split] = writer.class_counts

    write_dataset_metadata(
        dataset_root,
        policy,
        source={
            "adapter_id": policy.adapter_id,
            "source_root": str(source.root),
            "format": source.layout_id,
            "split_policy": "official_trials_to_train_test",
        },
        preprocessing=event_preprocessing_metadata(
            policy_id=policy.policy_id,
            num_steps=policy.num_steps,
            reduction="event_count",
            interval_locator=_SELECTION_INTERVAL_LOCATOR_ID,
            fixed_interval_seconds=(
                0.0,
                DVS128_GESTURE_PREPROCESSING_PROFILE.duration_s,
            ),
            spatial_shape=(128, 128),
            source_details={
                "locator_counts": locator_counts,
                "duration_us": DVS128_GESTURE_PREPROCESSING_PROFILE.duration_us,
                "bin_width_us": DVS128_GESTURE_PREPROCESSING_PROFILE.bin_width_us,
                "event_block_bytes": _AEDAT_EVENT_BLOCK_BYTES,
            },
        ),
        split_counts=counts,
        split_class_counts=split_class_counts,
    )


def _try_aedat_layout(source: Path) -> GestureSource | None:
    roots = (source / "DvsGesture", source)
    for root in roots:
        train_list = root / "trials_to_train.txt"
        test_list = root / "trials_to_test.txt"
        if not train_list.is_file() or not test_list.is_file():
            continue
        trial_names_by_split = {
            "train": _read_trial_list(train_list),
            "test": _read_trial_list(test_list),
        }
        samples: dict[str, tuple[GestureSample, ...]] = {}
        for split in ("train", "test"):
            split_samples: list[GestureSample] = []
            trial_names = trial_names_by_split[split]
            for trial_name in trial_names:
                trial_path = root / trial_name
                stem = Path(trial_name).stem
                labels_path = root / f"{stem}_labels.csv"
                rows = _read_label_rows(labels_path)
                per_label_ordinal = [0] * 11
                for row_index, (label, start, end) in enumerate(rows):
                    ordinal = per_label_ordinal[label]
                    per_label_ordinal[label] += 1
                    split_samples.append(
                        GestureSample(
                            split=split,
                            label=label,
                            sample_id=(
                                f"dvs128-gesture:{split}:{stem}:"
                                f"label{label:02d}:sample{ordinal:03d}:row{row_index:03d}"
                            ),
                            source_path=labels_path,
                            interval_start=start,
                            interval_end=end,
                            trial_path=trial_path,
                            row_index=row_index,
                        )
                    )
            samples[split] = tuple(split_samples)
        return GestureSource(
            root=root,
            layout_id="official_aedat_v3_csv",
            samples=samples,
        )
    return None


def _sample_intervals(
    sample: GestureSample,
) -> tuple[ObservationInterval, ObservationInterval]:
    if sample.interval_start is None or sample.interval_end is None:
        raise ValueError(
            f"gesture sample has no annotation interval: {sample.sample_id}"
        )
    start = int(sample.interval_start)
    annotation_end = int(sample.interval_end)
    fixed_end = start + DVS128_GESTURE_PREPROCESSING_PROFILE.duration_us
    support_interval = ObservationInterval(
        start=start,
        end=fixed_end,
        locator_id=_FIXED_INTERVAL_LOCATOR_ID,
    )
    selection_interval = ObservationInterval(
        start=start,
        end=min(annotation_end, fixed_end),
        locator_id=_SELECTION_INTERVAL_LOCATOR_ID,
    )
    return support_interval, selection_interval


def _consume_aedat_v3_header(handle: BinaryIO, *, path: Path) -> None:
    while True:
        position = handle.tell()
        line = handle.readline()
        if not line.startswith(b"#"):
            handle.seek(position)
            return
        if line.rstrip(b"\r\n") == b"#!END-HEADER":
            return


def _index_aedat_v3_polarity_packets(path: Path) -> tuple[_PolarityPacketIndex, ...]:
    packet_index: list[_PolarityPacketIndex] = []
    with path.open("rb") as handle:
        _consume_aedat_v3_header(handle, path=path)
        while True:
            header = handle.read(28)
            if not header:
                break
            event_type = struct.unpack_from("<H", header, 0)[0]
            event_size = struct.unpack_from("<I", header, 4)[0]
            ts_overflow = struct.unpack_from("<I", header, 12)[0]
            event_capacity = struct.unpack_from("<I", header, 16)[0]
            event_number = struct.unpack_from("<I", header, 20)[0]
            data_offset = handle.tell()
            data_length = event_capacity * event_size
            if event_type != 1:
                handle.seek(data_length, 1)
                continue

            min_timestamp: int | None = None
            max_timestamp: int | None = None
            for addresses, timestamps_low in _read_polarity_event_blocks(
                handle,
                path=path,
                event_size=event_size,
                event_number=event_number,
            ):
                valid = addresses & np.uint32(0x00000001) != 0
                valid_count = int(np.count_nonzero(valid))
                if valid_count == 0:
                    continue
                timestamps = timestamps_low[valid].astype(np.uint64)
                timestamps |= np.uint64(ts_overflow << 31)
                block_min = int(timestamps.min())
                block_max = int(timestamps.max())
                min_timestamp = (
                    block_min
                    if min_timestamp is None
                    else min(min_timestamp, block_min)
                )
                max_timestamp = (
                    block_max
                    if max_timestamp is None
                    else max(max_timestamp, block_max)
                )
            remaining_slots = event_capacity - event_number
            if remaining_slots:
                handle.seek(remaining_slots * event_size, 1)
            packet_index.append(
                _PolarityPacketIndex(
                    data_offset=data_offset,
                    event_size=event_size,
                    event_number=event_number,
                    ts_overflow=ts_overflow,
                    min_timestamp=min_timestamp,
                    max_timestamp=max_timestamp,
                )
            )
    return tuple(packet_index)


def _accumulate_gesture_segment(
    path: Path,
    *,
    packet_index: tuple[_PolarityPacketIndex, ...],
    support_interval: ObservationInterval,
    selection_interval: ObservationInterval,
    num_steps: int,
) -> np.ndarray:
    output = np.zeros((num_steps, 128, 128, 2), dtype=np.float64)
    for addresses_selected, timestamps_selected in _iter_gesture_segment_event_blocks(
        path,
        packet_index=packet_index,
        interval=selection_interval,
    ):
        time_index = fixed_width_duration_indices(
            timestamps_selected,
            interval=support_interval,
            num_steps=num_steps,
            bin_width=DVS128_GESTURE_PREPROCESSING_PROFILE.bin_width_us,
        )
        x = ((addresses_selected >> np.uint32(17)) & np.uint32(0x7FFF)).astype(np.int64)
        y = ((addresses_selected >> np.uint32(2)) & np.uint32(0x7FFF)).astype(np.int64)
        polarity = ((addresses_selected >> np.uint32(1)) & np.uint32(0x1)).astype(
            np.int64
        )
        np.add.at(output, (time_index, y, x, polarity), np.float64(1.0))
    return output


def _iter_gesture_segment_event_blocks(
    path: Path,
    *,
    packet_index: tuple[_PolarityPacketIndex, ...],
    interval: ObservationInterval,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    with path.open("rb") as handle:
        for packet in packet_index:
            if packet.min_timestamp is None or packet.max_timestamp is None:
                continue
            if (
                packet.min_timestamp >= interval.end
                or packet.max_timestamp < interval.start
            ):
                continue
            handle.seek(packet.data_offset)
            for addresses, timestamps_low in _read_polarity_event_blocks(
                handle,
                path=path,
                event_size=packet.event_size,
                event_number=packet.event_number,
            ):
                valid = addresses & np.uint32(0x00000001) != 0
                if not np.any(valid):
                    continue
                addresses_valid = addresses[valid]
                timestamps = timestamps_low[valid].astype(np.uint64)
                timestamps |= np.uint64(packet.ts_overflow << 31)
                selected = (timestamps >= interval.start) & (timestamps < interval.end)
                if not np.any(selected):
                    continue
                yield addresses_valid[selected], timestamps[selected]


def _read_polarity_event_blocks(
    handle: BinaryIO,
    *,
    path: Path,
    event_size: int,
    event_number: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    events_per_block = max(1, _AEDAT_EVENT_BLOCK_BYTES // event_size)
    remaining = int(event_number)
    while remaining:
        block_count = min(remaining, events_per_block)
        payload_size = block_count * event_size
        payload = handle.read(payload_size)
        addresses = np.ndarray(
            shape=(block_count,),
            dtype="<u4",
            buffer=payload,
            offset=0,
            strides=(event_size,),
        )
        timestamps = np.ndarray(
            shape=(block_count,),
            dtype="<u4",
            buffer=payload,
            offset=4,
            strides=(event_size,),
        )
        yield addresses, timestamps
        remaining -= block_count


def _read_trial_list(path: Path) -> tuple[str, ...]:
    names = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return names


def _read_label_rows(path: Path) -> tuple[tuple[int, int, int], ...]:
    rows: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = int(row["class"]) - 1
            start = int(row["startTime_usec"])
            end = int(row["endTime_usec"])
            rows.append((label, start, end))
    return tuple(rows)
