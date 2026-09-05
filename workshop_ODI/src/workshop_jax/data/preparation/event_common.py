from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ObservationInterval:
    start: int | float
    end: int | float
    locator_id: str


def fixed_width_duration_indices(
    timestamps: np.ndarray,
    *,
    interval: ObservationInterval,
    num_steps: int,
    bin_width: float,
) -> np.ndarray:

    values = np.asarray(timestamps).reshape(-1)
    if np.issubdtype(values.dtype, np.integer) and isinstance(
        bin_width, (int, np.integer)
    ):
        relative = values.astype(np.int64) - int(interval.start)
        return relative // int(bin_width)
    relative_float = values.astype(np.float64) - float(interval.start)
    return np.floor(relative_float / float(bin_width)).astype(np.int64)


def fixed_duration_event_mask(
    timestamps: np.ndarray,
    *,
    interval: ObservationInterval,
) -> np.ndarray:

    values = np.asarray(timestamps).reshape(-1)
    return (values >= interval.start) & (values < interval.end)


def accumulate_fixed_window_pooled_channel_counts(
    *,
    timestamps_seconds: np.ndarray,
    channels: np.ndarray,
    interval: ObservationInterval,
    num_steps: int,
    bin_width_seconds: float,
    raw_num_channels: int,
    channel_pool_size: int,
) -> np.ndarray:

    timestamp_values = np.asarray(timestamps_seconds, dtype=np.float64).reshape(-1)
    channel_values = np.asarray(channels, dtype=np.int64).reshape(-1)
    valid = fixed_duration_event_mask(timestamp_values, interval=interval)
    output_channels = raw_num_channels // channel_pool_size
    output = np.zeros((num_steps, output_channels), dtype=np.float64)
    if not np.any(valid):
        return output
    valid_timestamps = timestamp_values[valid]
    valid_channels = channel_values[valid]
    time_indices = fixed_width_duration_indices(
        valid_timestamps,
        interval=interval,
        num_steps=num_steps,
        bin_width=bin_width_seconds,
    )
    pooled_channels = valid_channels // channel_pool_size
    flat_indices = np.ravel_multi_index(
        (time_indices, pooled_channels),
        output.shape,
    )
    np.add.at(output.reshape(-1), flat_indices, np.float64(1.0))
    return output


def accumulate_fixed_window_spatial_event_counts(
    *,
    timestamps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    polarity: np.ndarray,
    interval: ObservationInterval,
    num_steps: int,
    bin_width: int,
    height: int,
    width: int,
) -> np.ndarray:

    timestamp_values = np.asarray(timestamps).reshape(-1)
    valid = fixed_duration_event_mask(timestamp_values, interval=interval)
    output = np.zeros((num_steps, height, width, 2), dtype=np.float64)
    if not np.any(valid):
        return output
    time_indices = fixed_width_duration_indices(
        timestamp_values[valid],
        interval=interval,
        num_steps=num_steps,
        bin_width=bin_width,
    )
    x_values = np.asarray(x, dtype=np.int64).reshape(-1)[valid]
    y_values = np.asarray(y, dtype=np.int64).reshape(-1)[valid]
    p_values = np.asarray(polarity, dtype=np.int64).reshape(-1)[valid]
    np.add.at(
        output,
        (time_indices, y_values, x_values, p_values),
        np.float64(1.0),
    )
    return output


def event_preprocessing_metadata(
    *,
    policy_id: str,
    num_steps: int,
    reduction: str,
    interval_locator: str,
    fixed_interval_seconds: tuple[float, float],
    spatial_shape: tuple[int, int] | None = None,
    source_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "num_steps": num_steps,
        "boundary_rule": "left_closed_right_open",
        "reduction": reduction,
        "floating_compute_dtype": "float64",
        "storage_dtype": "float64",
        "interval_locator": interval_locator,
        "interval_seconds": [float(value) for value in fixed_interval_seconds],
        "spatial_shape": list(spatial_shape) if spatial_shape is not None else None,
        "source": source_details or {},
    }


__all__ = [
    "ObservationInterval",
    "accumulate_fixed_window_pooled_channel_counts",
    "accumulate_fixed_window_spatial_event_counts",
    "event_preprocessing_metadata",
    "fixed_duration_event_mask",
    "fixed_width_duration_indices",
]
