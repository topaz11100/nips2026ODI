from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from workshop_jax.analysis.spectral import layer_mean_distance
from workshop_jax.models import regularization_apply
from workshop_jax.models.readouts import apply_readout
from workshop_jax.models.spatial import BATCH_NORM_EPSILON
from workshop_jax.training.steps import _metric_sums


class _CapturedMoments(Exception):
    def __init__(self, value: Any):
        self.value = value


def normalization_moments(values: Any) -> Any:
    x = jnp.asarray(values, dtype=jnp.float64)
    axes = tuple(range(x.ndim - 1))
    return jnp.stack((jnp.mean(x, axis=axes), jnp.mean(jnp.square(x), axis=axes)))


def normalize_with_moments(values: Any, scale: Any, beta: Any, moments: Any) -> Any:
    x = jnp.asarray(values)
    mean = moments[0]
    variance = jnp.maximum(moments[1] - jnp.square(mean), 0.0)
    normalized = (x - mean.astype(x.dtype)) * jax.lax.rsqrt(
        variance.astype(x.dtype) + jnp.asarray(BATCH_NORM_EPSILON, dtype=x.dtype)
    )
    return normalized * scale + beta


def calibration_apply(
    params: Any,
    model_state: Any,
    inputs: Any,
    *,
    plan: Any,
    statistics: Any = None,
    capture_at: int | None = None,
    apply: Callable[..., Any] = regularization_apply,
) -> Any:
    if "normalization" not in model_state:
        with jax.default_matmul_precision("highest"):
            return apply(params, model_state, inputs, plan=plan)
    site = 0

    def normalize(x: Any, scale: Any, beta: Any, state: Any) -> Any:
        nonlocal site
        index = site
        site += 1
        if index == capture_at:
            raise _CapturedMoments(normalization_moments(x))
        moments = normalization_moments(x) if statistics is None else statistics[index]
        return normalize_with_moments(x, scale, beta, moments), state

    try:
        with jax.default_matmul_precision("highest"):
            result = apply(
                params, model_state, inputs, plan=plan, normalization=normalize
            )
    except _CapturedMoments as captured:
        return captured.value
    if capture_at is not None:
        raise ValueError("normalization site was not reached")
    return result


def full_batch_spectrum(signal: Any) -> Any:
    x = jnp.asarray(signal, dtype=jnp.float32)
    x = x.reshape((x.shape[0], x.shape[1], -1))
    centered = x - jnp.mean(x, axis=1, keepdims=True)
    magnitude = jnp.abs(jnp.fft.rfft(centered, axis=1))
    return jnp.mean(magnitude, axis=(0, 2), dtype=jnp.float64)


def batch_summaries(
    params: Any,
    model_state: Any,
    batch: Any,
    *,
    plan: Any,
    statistics: Any = None,
    apply: Callable[..., Any] = regularization_apply,
) -> Any:
    applied = calibration_apply(
        params, model_state, batch.inputs, plan=plan, statistics=statistics, apply=apply
    )
    metrics = _metric_sums(apply_readout(applied.output, plan.readout), batch, plan)
    task = metrics.loss_sum.astype(jnp.float64) / metrics.count
    hidden = tuple(applied.traces["spike_output"])[:-1]
    if len(hidden) != len(plan.hidden_sizes):
        raise ValueError("hidden trace count does not match the model plan")
    return (
        task,
        full_batch_spectrum(batch.inputs),
        tuple(full_batch_spectrum(value) for value in hidden),
    )


def full_batch_losses(
    params: Any,
    model_state: Any,
    batch: Any,
    *,
    plan: Any,
    apply: Callable[..., Any] = regularization_apply,
) -> Any:
    task, reference, hidden = batch_summaries(
        params, model_state, batch, plan=plan, apply=apply
    )
    return jnp.stack((task, -layer_mean_distance(reference, hidden)))


def _add(left: Any, right: Any) -> Any:
    return jax.tree.map(lambda a, b: a + b, left, right)


def _norm(tree: Any) -> float:
    return float(
        np.sqrt(
            sum(
                np.sum(np.square(value), dtype=np.float64)
                for value in jax.tree.leaves(tree)
            )
        )
    )


@dataclass(frozen=True)
class CalibrationResult:
    task_gradient: Any
    smr_gradient: Any
    sample_count: int
    task_loss: float
    distance: float
    normalization_statistics: Any

    def coefficient(self, target_ratio: float) -> float:
        if not np.isfinite(target_ratio) or target_ratio <= 0:
            raise ValueError("target ratio must be positive and finite")
        task_norm = _norm(self.task_gradient["hidden"])
        smr_norm = _norm(self.smr_gradient["hidden"])
        if not np.isfinite(task_norm) or not np.isfinite(smr_norm):
            raise FloatingPointError("nonfinite full-batch gradient norm")
        if task_norm == 0 or smr_norm == 0:
            raise ValueError(
                "zero full-batch gradient norm; a positive target ratio is undefined"
            )
        return target_ratio * task_norm / smr_norm


def calibrate_full_batch(
    params: Any,
    model_state: Any,
    *,
    plan: Any,
    batches: Callable[[], Iterable[Any]],
    sample_count: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
    apply: Callable[..., Any] = regularization_apply,
) -> CalibrationResult:
    if not jax.config.x64_enabled:
        raise RuntimeError("FP64 is required for full-batch accumulation")
    started = time.monotonic()

    def average(kernel: Callable[[Any], Any], phase: str) -> Any:
        total = None
        seen = 0
        if progress is not None:
            progress({"phase": phase, "samples_seen": 0, "samples_total": sample_count})
        for index, batch in enumerate(batches(), start=1):
            mask = np.asarray(jax.device_get(batch.real_sample_mask))
            if not mask.all():
                raise ValueError("full-batch calibration requires unpadded minibatches")
            count = len(mask)
            values = jax.device_get(kernel(batch))
            if not all(np.isfinite(value).all() for value in jax.tree.leaves(values)):
                raise FloatingPointError(f"nonfinite values in {phase} batch {index}")
            weighted = jax.tree.map(
                lambda value: np.asarray(value, dtype=np.float64) * count, values
            )
            total = weighted if total is None else _add(total, weighted)
            seen += count
            if progress is not None and (
                index == 1 or index % 20 == 0 or seen == sample_count
            ):
                progress(
                    {
                        "phase": phase,
                        "batch": index,
                        "samples_seen": seen,
                        "samples_total": sample_count,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
        if total is None or seen != sample_count:
            raise ValueError(f"{phase} did not cover the complete calibration set")
        return jax.tree.map(lambda value: value / sample_count, total)

    normalization_count = len(model_state.get("normalization", {}).get("hidden", ()))
    statistics: tuple[Any, ...] = ()
    for site in range(normalization_count):
        capture: Any = jax.jit(
            lambda p, s, batch: calibration_apply(
                p,
                model_state,
                batch.inputs,
                plan=plan,
                statistics=s,
                capture_at=site,
                apply=apply,
            )
        )
        value = average(
            lambda batch: capture(params, statistics, batch),
            f"normalization_forward_{site + 1}/{normalization_count}",
        )
        statistics = (*statistics, jax.device_put(value))
        capture.clear_cache()

    summaries: Any = jax.jit(
        lambda p, s, batch: batch_summaries(
            p, model_state, batch, plan=plan, statistics=s, apply=apply
        )
    )
    task_loss, reference, hidden = average(
        lambda batch: summaries(params, statistics, batch), "global_spectra"
    )
    curve_cotangent = jax.grad(lambda curves: -layer_mean_distance(reference, curves))(
        hidden
    )
    distance = float(layer_mean_distance(reference, hidden))
    summaries.clear_cache()

    def task_objective(p: Any, s: Any, batch: Any) -> Any:
        return batch_summaries(
            p, model_state, batch, plan=plan, statistics=s, apply=apply
        )[0]

    def smr_objective(p: Any, s: Any, batch: Any) -> Any:
        curves = batch_summaries(
            p, model_state, batch, plan=plan, statistics=s, apply=apply
        )[2]
        return sum(
            jnp.vdot(value, cotangent)
            for value, cotangent in zip(curves, curve_cotangent, strict=True)
        )

    task_kernel: Any = jax.jit(jax.grad(task_objective, argnums=(0, 1)))
    smr_kernel: Any = jax.jit(jax.grad(smr_objective, argnums=(0, 1)))

    def direct(batch: Any) -> Any:
        task = jax.device_get(task_kernel(params, statistics, batch))
        smr = jax.device_get(smr_kernel(params, statistics, batch))
        return task, smr

    (task_gradient, task_statistics), (smr_gradient, smr_statistics) = average(
        direct, "direct_gradients"
    )
    task_kernel.clear_cache()
    smr_kernel.clear_cache()

    for site in reversed(range(normalization_count)):
        task_cotangent = jax.device_put(task_statistics[site])
        smr_cotangent = jax.device_put(smr_statistics[site])
        if not np.any(task_statistics[site]) and not np.any(smr_statistics[site]):
            continue

        def objective(p: Any, s: Any, batch: Any, cotangent: Any) -> Any:
            moments = calibration_apply(
                p,
                model_state,
                batch.inputs,
                plan=plan,
                statistics=s,
                capture_at=site,
                apply=apply,
            )
            return jnp.vdot(moments, cotangent)

        kernel: Any = jax.jit(jax.grad(objective, argnums=(0, 1)))

        def indirect(batch: Any) -> Any:
            task = jax.device_get(kernel(params, statistics, batch, task_cotangent))
            smr = jax.device_get(kernel(params, statistics, batch, smr_cotangent))
            return task, smr

        (task_extra, task_stats_extra), (smr_extra, smr_stats_extra) = average(
            indirect, f"normalization_backward_{site + 1}/{normalization_count}"
        )
        task_gradient = _add(task_gradient, task_extra)
        smr_gradient = _add(smr_gradient, smr_extra)
        task_statistics = _add(task_statistics, task_stats_extra)
        smr_statistics = _add(smr_statistics, smr_stats_extra)
        kernel.clear_cache()
    return CalibrationResult(
        task_gradient,
        smr_gradient,
        sample_count,
        float(task_loss),
        distance,
        statistics,
    )
