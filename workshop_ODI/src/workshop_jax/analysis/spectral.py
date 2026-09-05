from __future__ import annotations

from typing import Any, Sequence

import jax.numpy as jnp


def _curve_sum(
    curve: Any,
    real_sample_mask: Any,
    component_count: int,
    *,
    dtype: Any,
) -> tuple[Any, Any]:
    mask = jnp.asarray(real_sample_mask, dtype=dtype)
    weighted = jnp.asarray(curve, dtype=dtype) * mask[:, None, None]
    total = jnp.sum(weighted, axis=(0, 2), dtype=dtype)
    count = jnp.sum(mask, dtype=dtype) * jnp.asarray(component_count, dtype=dtype)
    return total, count


def _dft_magnitude_sum(
    signal: Any,
    real_sample_mask: Any,
    *,
    dtype: Any,
) -> tuple[Any, Any]:
    values = jnp.asarray(signal, dtype=dtype)
    values = values.reshape((values.shape[0], values.shape[1], -1))
    if int(values.shape[1]) <= 1:
        raise ValueError("DFT analysis requires at least two time steps")
    centered = values - jnp.mean(values, axis=1, keepdims=True, dtype=dtype)
    length = int(centered.shape[1])
    spectrum = jnp.fft.rfft(centered, n=length, axis=1)
    magnitude = jnp.abs(spectrum).astype(dtype)
    return _curve_sum(
        magnitude,
        real_sample_mask,
        int(values.shape[2]),
        dtype=dtype,
    )


def dft_magnitude_sum(signal: Any, real_sample_mask: Any) -> tuple[Any, Any]:
    return _dft_magnitude_sum(signal, real_sample_mask, dtype=jnp.float64)


def _representative_dft(
    signal: Any,
    real_sample_mask: Any,
    *,
    dtype: Any,
) -> Any:
    total, count = _dft_magnitude_sum(
        signal,
        real_sample_mask,
        dtype=dtype,
    )
    return total / jnp.maximum(count, jnp.asarray(1.0, dtype=dtype))


def representative_dft(signal: Any, real_sample_mask: Any) -> Any:
    return _representative_dft(signal, real_sample_mask, dtype=jnp.float64)


def _centered_l2(reference: Any, estimate: Any, *, dtype: Any) -> Any:
    left = jnp.asarray(reference, dtype=dtype)
    right = jnp.asarray(estimate, dtype=dtype)
    left = left - jnp.mean(left, dtype=dtype)
    right = right - jnp.mean(right, dtype=dtype)
    squared = jnp.sum(jnp.square(left - right), dtype=dtype)
    positive = squared > 0.0
    norm = jnp.sqrt(jnp.where(positive, squared, jnp.ones_like(squared)))
    return jnp.where(positive, norm, jnp.zeros_like(norm)).astype(dtype)


def centered_l2(reference: Any, estimate: Any) -> Any:
    return _centered_l2(reference, estimate, dtype=jnp.float64)


def _layer_mean_distance(
    reference: Any,
    estimates: Sequence[Any],
    *,
    dtype: Any,
) -> Any:
    if not estimates:
        raise ValueError("D_model requires at least one estimate layer")
    distances = jnp.stack(
        tuple(_centered_l2(reference, estimate, dtype=dtype) for estimate in estimates)
    )
    return jnp.mean(distances, dtype=dtype)


def layer_mean_distance(reference: Any, estimates: Sequence[Any]) -> Any:
    return _layer_mean_distance(reference, estimates, dtype=jnp.float64)


def batch_model_distance(
    dataset_inputs: Any,
    hidden_spikes: Sequence[Any],
    real_sample_mask: Any,
) -> Any:
    signals = tuple(hidden_spikes)
    if not signals:
        raise ValueError("D_model requires at least one estimate layer")
    reference = _representative_dft(
        dataset_inputs,
        real_sample_mask,
        dtype=jnp.float32,
    )
    estimates = tuple(
        _representative_dft(signal, real_sample_mask, dtype=jnp.float32)
        for signal in signals
    )
    return _layer_mean_distance(reference, estimates, dtype=jnp.float32)


__all__ = [
    "batch_model_distance",
    "centered_l2",
    "dft_magnitude_sum",
    "layer_mean_distance",
    "representative_dft",
]
