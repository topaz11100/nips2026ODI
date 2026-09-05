from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp

from workshop_jax.models.specification import ModelOutput
from workshop_jax.neurons.lif import LIF


BATCH_NORM_EPSILON = 1.0e-5
BATCH_NORM_MOMENTUM = 0.1


def conv2d_time(
    inputs: Any,
    weight: Any,
    bias: Any | None = None,
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> Any:

    return _conv2d_time_impl(
        inputs,
        weight,
        bias,
        stride=stride,
        padding=padding,
    )


def _conv2d_time_impl(
    inputs: Any,
    weight: Any,
    bias: Any | None,
    *,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> Any:

    kernel = jnp.asarray(weight)
    x = jnp.asarray(inputs, dtype=kernel.dtype)
    stride_pair = _pair(stride)
    padding_pair = _pair(padding)
    flattened = x.reshape((-1, *x.shape[2:]))
    convolved = jax.lax.conv_general_dilated(
        flattened,
        kernel,
        window_strides=stride_pair,
        padding=(
            (padding_pair[0], padding_pair[0]),
            (padding_pair[1], padding_pair[1]),
        ),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        precision=None,
    )
    if bias is not None:
        convolved = (
            convolved + jnp.asarray(bias, dtype=convolved.dtype)[None, None, None, :]
        )
    return convolved.reshape((x.shape[0], x.shape[1], *convolved.shape[1:]))


def linear_time(
    inputs: Any,
    weight: Any,
    bias: Any | None = None,
) -> Any:

    return _linear_time_impl(inputs, weight, bias)


def _linear_time_impl(inputs: Any, weight: Any, bias: Any | None) -> Any:

    kernel = jnp.asarray(weight)
    output = jnp.einsum(
        "...i,io->...o",
        jnp.asarray(inputs, dtype=kernel.dtype),
        kernel,
        precision=None,
    )
    if bias is not None:
        output = output + jnp.asarray(bias, dtype=output.dtype)
    return output


def average_pool2d_time(
    inputs: Any,
    *,
    kernel: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int] = 0,
) -> Any:
    x = jnp.asarray(inputs)
    kernel_pair = _pair(kernel)
    stride_pair = _pair(stride)
    padding_pair = _pair(padding)
    flattened = x.reshape((-1, *x.shape[2:]))
    channels = int(flattened.shape[-1])
    denominator = jnp.asarray(kernel_pair[0] * kernel_pair[1], dtype=x.dtype)
    depthwise_kernel = (
        jnp.ones((kernel_pair[0], kernel_pair[1], 1, channels), dtype=x.dtype)
        / denominator
    )
    pooled = jax.lax.conv_general_dilated(
        flattened,
        depthwise_kernel,
        window_strides=stride_pair,
        padding=(
            (padding_pair[0], padding_pair[0]),
            (padding_pair[1], padding_pair[1]),
        ),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=channels,
        precision=None,
    )
    return pooled.reshape((x.shape[0], x.shape[1], *pooled.shape[1:]))


def max_pool2d_time(
    inputs: Any,
    *,
    kernel: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int] = 0,
) -> Any:
    x = jnp.asarray(inputs)
    kernel_pair = _pair(kernel)
    stride_pair = _pair(stride)
    padding_pair = _pair(padding)
    flattened = x.reshape((-1, *x.shape[2:]))
    pooled = jax.lax.reduce_window(
        operand=flattened,
        init_value=-jnp.inf,
        computation=jax.lax.max,
        window_dimensions=(1, kernel_pair[0], kernel_pair[1], 1),
        window_strides=(1, stride_pair[0], stride_pair[1], 1),
        padding=(
            (0, 0),
            (padding_pair[0], padding_pair[0]),
            (padding_pair[1], padding_pair[1]),
            (0, 0),
        ),
    )
    return pooled.reshape((x.shape[0], x.shape[1], *pooled.shape[1:]))


def batch_norm_train_time(
    inputs: Any,
    scale: Any,
    beta: Any,
    state: Mapping[str, Any],
    *,
    epsilon: float = BATCH_NORM_EPSILON,
    momentum: float = BATCH_NORM_MOMENTUM,
) -> tuple[Any, dict[str, Any]]:

    scale_array = jnp.asarray(scale)
    x = jnp.asarray(inputs, dtype=scale_array.dtype)
    axes = tuple(range(x.ndim - 1))
    running_mean = jnp.asarray(state["running_mean"], dtype=x.dtype)
    running_variance = jnp.asarray(state["running_variance"], dtype=x.dtype)
    mean = jnp.mean(x, axis=axes)
    variance = jnp.mean(jnp.square(x - mean), axis=axes)
    population = math.prod(int(x.shape[axis]) for axis in axes)
    correction = 1.0 if population <= 1 else float(population) / float(population - 1)
    next_state = {
        "running_mean": (1.0 - momentum) * running_mean + momentum * mean,
        "running_variance": (
            (1.0 - momentum) * running_variance
            + momentum * variance * jnp.asarray(correction, dtype=variance.dtype)
        ),
    }
    normalized = (x - mean) * jax.lax.rsqrt(
        variance + jnp.asarray(epsilon, dtype=x.dtype)
    )
    return normalized * scale_array + jnp.asarray(beta, dtype=x.dtype), next_state


def batch_norm_frozen_time(
    inputs: Any,
    scale: Any,
    beta: Any,
    state: Mapping[str, Any],
    *,
    epsilon: float = BATCH_NORM_EPSILON,
) -> Any:

    scale_array = jnp.asarray(scale)
    x = jnp.asarray(inputs, dtype=scale_array.dtype)
    running_mean = jnp.asarray(state["running_mean"], dtype=x.dtype)
    running_variance = jnp.asarray(state["running_variance"], dtype=x.dtype)
    normalized = (x - running_mean) * jax.lax.rsqrt(
        running_variance + jnp.asarray(epsilon, dtype=x.dtype)
    )
    return normalized * scale_array + jnp.asarray(beta, dtype=x.dtype)


def init_batch_norm_state(channels: int) -> dict[str, Any]:
    return {
        "running_mean": jnp.zeros((channels,), dtype=jnp.float32),
        "running_variance": jnp.ones((channels,), dtype=jnp.float32),
    }


def init_kaiming_normal_fan_out(
    key: Any,
    shape: Sequence[int],
) -> Any:
    shape_tuple = tuple(int(value) for value in shape)
    if len(shape_tuple) == 4:
        fan_out = shape_tuple[0] * shape_tuple[1] * shape_tuple[3]
    elif len(shape_tuple) == 2:
        fan_out = shape_tuple[1]
    else:
        raise ValueError(f"unsupported fan-out initializer shape: {shape_tuple}")
    std = math.sqrt(2.0 / float(fan_out))
    return jax.random.normal(key, shape_tuple, dtype=jnp.float32) * jnp.asarray(
        std, dtype=jnp.float32
    )


def init_normal(key: Any, shape: Sequence[int], *, std: float) -> Any:
    return jax.random.normal(
        key, tuple(int(value) for value in shape), dtype=jnp.float32
    ) * jnp.asarray(float(std), dtype=jnp.float32)


def run_spatial_neuron_site(
    unit_current: Any,
    neuron_params: Mapping[str, Any],
    neuron_fixed: Mapping[str, Any],
    *,
    plan: Any,
) -> Any:
    current = jnp.asarray(unit_current)
    batch_size = int(current.shape[0])
    sequence_length = int(current.shape[1])
    spatial_shape = tuple(int(value) for value in current.shape[2:-1])
    channels = int(current.shape[-1])
    coordinate_count = math.prod(spatial_shape)
    module = LIF
    state = module.init_state(
        plan.neuron,
        batch_size=batch_size * coordinate_count,
        units=channels,
        dtype=current.dtype,
    )
    time_major = jnp.swapaxes(
        current.reshape((batch_size, sequence_length, coordinate_count, channels)),
        0,
        1,
    )

    def body(carry: Any, current_t: Any) -> tuple[Any, Any]:
        current_flat = current_t.reshape((batch_size * coordinate_count, channels))
        neuron_current = jnp.asarray(current_flat)
        result = module.step(
            plan.neuron,
            neuron_params,
            neuron_fixed,
            carry,
            neuron_current,
        )
        spike_output = result.spike_output.reshape(
            (batch_size, *spatial_shape, channels)
        )
        return result.state, spike_output

    _, output_time = jax.lax.scan(
        body,
        state,
        time_major,
        unroll=plan.scan_unroll,
    )
    return jnp.swapaxes(output_time, 0, 1)


def run_output_neuron_site(
    class_current: Any,
    neuron_params: Mapping[str, Any],
    neuron_fixed: Mapping[str, Any],
    *,
    plan: Any,
) -> ModelOutput:
    current = jnp.asarray(class_current)
    batch_size, _, classes = map(int, current.shape)
    module = LIF
    actual_state = module.init_state(
        plan.neuron,
        batch_size=batch_size,
        units=classes,
        dtype=current.dtype,
    )
    linear_state = module.init_state(
        plan.neuron,
        batch_size=batch_size,
        units=classes,
        dtype=current.dtype,
    )
    time_major = jnp.swapaxes(current, 0, 1)

    def body(carry: Any, current_t: Any) -> tuple[Any, Any]:
        actual, linear = carry
        neuron_current = jnp.asarray(current_t)
        result = module.step(
            plan.neuron,
            neuron_params,
            neuron_fixed,
            actual,
            neuron_current,
        )
        next_linear, reset_free = module.linear_step(
            plan.neuron,
            neuron_params,
            neuron_fixed,
            linear,
            neuron_current,
        )
        return (result.state, next_linear), (result.spike_output, reset_free)

    _, (spikes_time, membrane_time) = jax.lax.scan(
        body,
        (actual_state, linear_state),
        time_major,
        unroll=plan.scan_unroll,
    )
    return ModelOutput(
        spike_output=jnp.swapaxes(spikes_time, 0, 1),
        membrane_pre=jnp.swapaxes(membrane_time, 0, 1),
    )


def _restore_spatial_coordinate(
    value: Any,
    *,
    batch_size: int,
    spatial_shape: Sequence[int],
) -> Any:

    array = jnp.asarray(value)
    canonical_spatial_shape = tuple(int(size) for size in spatial_shape)
    return array.reshape((batch_size, *canonical_spatial_shape, *array.shape[1:]))


def run_spatial_neuron_site_signal(
    unit_current: Any,
    neuron_params: Mapping[str, Any],
    neuron_fixed: Mapping[str, Any],
    *,
    plan: Any,
) -> tuple[Any, dict[str, Any]]:

    current = jnp.asarray(unit_current)
    batch_size = int(current.shape[0])
    sequence_length = int(current.shape[1])
    spatial_shape = tuple(int(value) for value in current.shape[2:-1])
    channels = int(current.shape[-1])
    coordinate_count = math.prod(spatial_shape)
    module = LIF
    state = module.init_state(
        plan.neuron,
        batch_size=batch_size * coordinate_count,
        units=channels,
        dtype=current.dtype,
    )
    time_major = jnp.swapaxes(
        current.reshape((batch_size, sequence_length, coordinate_count, channels)),
        0,
        1,
    )

    def body(carry: Any, current_t: Any) -> tuple[Any, Any]:
        current_flat = current_t.reshape((batch_size * coordinate_count, channels))
        neuron_current = jnp.asarray(current_flat)
        result = module.step(
            plan.neuron,
            neuron_params,
            neuron_fixed,
            carry,
            neuron_current,
        )
        trace_output = {
            "spike_output": _restore_spatial_coordinate(
                result.spike_output,
                batch_size=batch_size,
                spatial_shape=spatial_shape,
            )
        }
        spike_output = result.spike_output.reshape(
            (batch_size, *spatial_shape, channels)
        )
        return result.state, (spike_output, trace_output)

    _, (output_time, trace_time) = jax.lax.scan(
        body,
        state,
        time_major,
        unroll=plan.scan_unroll,
    )
    return (
        jnp.swapaxes(output_time, 0, 1),
        {signal: jnp.swapaxes(value, 0, 1) for signal, value in trace_time.items()},
    )


def run_output_neuron_site_signal(
    class_current: Any,
    neuron_params: Mapping[str, Any],
    neuron_fixed: Mapping[str, Any],
    *,
    plan: Any,
) -> tuple[ModelOutput, dict[str, Any]]:

    current = jnp.asarray(class_current)
    batch_size, _, classes = map(int, current.shape)
    module = LIF
    actual_state = module.init_state(
        plan.neuron,
        batch_size=batch_size,
        units=classes,
        dtype=current.dtype,
    )
    linear_state = module.init_state(
        plan.neuron,
        batch_size=batch_size,
        units=classes,
        dtype=current.dtype,
    )
    time_major = jnp.swapaxes(current, 0, 1)

    def body(carry: Any, current_t: Any) -> tuple[Any, Any]:
        actual, linear = carry
        neuron_current = jnp.asarray(current_t)
        result = module.step(
            plan.neuron,
            neuron_params,
            neuron_fixed,
            actual,
            neuron_current,
        )
        next_linear, reset_free = module.linear_step(
            plan.neuron,
            neuron_params,
            neuron_fixed,
            linear,
            neuron_current,
        )
        return (result.state, next_linear), (
            result.spike_output,
            reset_free,
            {"spike_output": result.spike_output},
        )

    _, (spikes_time, membrane_time, trace_time) = jax.lax.scan(
        body,
        (actual_state, linear_state),
        time_major,
        unroll=plan.scan_unroll,
    )
    return (
        ModelOutput(
            spike_output=jnp.swapaxes(spikes_time, 0, 1),
            membrane_pre=jnp.swapaxes(membrane_time, 0, 1),
        ),
        {signal: jnp.swapaxes(value, 0, 1) for signal, value in trace_time.items()},
    )


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


__all__ = [
    "BATCH_NORM_EPSILON",
    "BATCH_NORM_MOMENTUM",
    "average_pool2d_time",
    "batch_norm_frozen_time",
    "batch_norm_train_time",
    "conv2d_time",
    "init_batch_norm_state",
    "init_kaiming_normal_fan_out",
    "init_normal",
    "linear_time",
    "max_pool2d_time",
    "run_output_neuron_site",
    "run_output_neuron_site_signal",
    "run_spatial_neuron_site",
    "run_spatial_neuron_site_signal",
]
