from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp

from workshop_jax.models.initialization_rng import InitializationKeyspace
from workshop_jax.models.specification import ApplyResult, ModelOutput, ModelPlan
from workshop_jax.neurons.lif import LIF
from workshop_jax.training.initialization import initialize_neuron_layer, kaiming_weight


def _initialize_projection(
    keyspace: InitializationKeyspace,
    site_path: tuple[str | int, ...],
    fan_in: int,
    units: int,
) -> dict[str, Any]:
    key = keyspace.topology_key((*site_path, "weight"))
    return {
        "weight": kaiming_weight(key, fan_in, units),
        "bias": jnp.zeros((units,), dtype=jnp.float32),
    }


def _apply_projection(source: Any, layer: Mapping[str, Any]) -> Any:
    return jnp.asarray(source) @ layer["weight"] + layer["bias"]


def module_init(
    key: Any,
    plan: ModelPlan,
) -> tuple[dict[str, Any], dict[str, Any]]:
    keyspace = InitializationKeyspace(key, topology_id=plan.topology_id)
    hidden_params: list[dict[str, Any]] = []
    hidden_fixed: list[dict[str, Any]] = []
    fan_in = plan.input_dim
    for index, width in enumerate(plan.hidden_sizes):
        site_path = ("hidden", index)
        projection = _initialize_projection(keyspace, site_path, fan_in, width)
        neuron_params, neuron_fixed = initialize_neuron_layer(
            plan.neuron,
            units=width,
        )
        hidden_params.append({**projection, "neuron": neuron_params})
        hidden_fixed.append(neuron_fixed)
        fan_in = width
    output_site_path = ("output",)
    output_projection = _initialize_projection(
        keyspace,
        output_site_path,
        fan_in,
        plan.num_classes,
    )
    output_neuron_params, output_neuron_fixed = initialize_neuron_layer(
        plan.neuron,
        units=plan.num_classes,
    )
    params = {
        "hidden": tuple(hidden_params),
        "output": {**output_projection, "neuron": output_neuron_params},
    }
    model_state = {
        "neuron_fixed": {
            "hidden": tuple(hidden_fixed),
            "output": output_neuron_fixed,
        },
    }
    return params, model_state


def module_train_apply(
    params: Mapping[str, Any],
    model_state: Mapping[str, Any],
    inputs: Any,
    *,
    plan: ModelPlan,
) -> ApplyResult:

    return _module_train_forward(
        params,
        model_state,
        inputs,
        plan=plan,
    )


def module_eval_apply(
    params: Mapping[str, Any],
    model_state: Mapping[str, Any],
    inputs: Any,
    *,
    plan: ModelPlan,
) -> ApplyResult:

    return _module_eval_forward(
        params,
        model_state,
        inputs,
        plan=plan,
    )


def _module_train_forward(
    params: Mapping[str, Any],
    model_state: Mapping[str, Any],
    inputs: Any,
    *,
    plan: ModelPlan,
) -> ApplyResult:
    module = LIF
    batch_size = int(inputs.shape[0])
    flattened = inputs.reshape(batch_size, inputs.shape[1], -1)

    time_major = jnp.swapaxes(flattened, 0, 1)
    hidden_state = tuple(
        module.init_state(
            plan.neuron,
            batch_size=batch_size,
            units=width,
            dtype=inputs.dtype,
        )
        for width in plan.hidden_sizes
    )
    output_state = module.init_state(
        plan.neuron,
        batch_size=batch_size,
        units=plan.num_classes,
        dtype=inputs.dtype,
    )
    output_linear_state = module.init_state(
        plan.neuron,
        batch_size=batch_size,
        units=plan.num_classes,
        dtype=inputs.dtype,
    )

    fixed_hidden = model_state["neuron_fixed"]["hidden"]
    fixed_output = model_state["neuron_fixed"]["output"]

    def scan_body(carry: Any, input_t: Any) -> tuple[Any, Any]:
        states, output_actual_state, output_free_state = carry
        next_states: list[Any] = []
        current = input_t

        for index in range(len(plan.hidden_sizes)):
            layer = params["hidden"][index]
            neuron_current = _apply_projection(current, layer)
            result = module.step(
                plan.neuron,
                layer["neuron"],
                fixed_hidden[index],
                states[index],
                neuron_current,
            )
            next_states.append(result.state)
            current = result.spike_output

        output_layer = params["output"]
        class_current = _apply_projection(current, output_layer)
        output_result = module.step(
            plan.neuron,
            output_layer["neuron"],
            fixed_output,
            output_actual_state,
            class_current,
        )
        next_output_free_state, reset_free_membrane = module.linear_step(
            plan.neuron,
            output_layer["neuron"],
            fixed_output,
            output_free_state,
            class_current,
        )
        next_carry = (
            tuple(next_states),
            output_result.state,
            next_output_free_state,
        )
        return next_carry, (output_result.spike_output, reset_free_membrane)

    _, (spikes_time, membrane_time) = jax.lax.scan(
        scan_body,
        (hidden_state, output_state, output_linear_state),
        time_major,
        unroll=plan.scan_unroll,
    )
    return ApplyResult(
        output=ModelOutput(
            spike_output=jnp.swapaxes(spikes_time, 0, 1),
            membrane_pre=jnp.swapaxes(membrane_time, 0, 1),
        ),
        model_state=model_state,
        traces={},
    )


def _module_eval_forward(
    params: Mapping[str, Any],
    model_state: Mapping[str, Any],
    inputs: Any,
    *,
    plan: ModelPlan,
) -> ApplyResult:
    module = LIF
    batch_size = int(inputs.shape[0])
    flattened = inputs.reshape(batch_size, inputs.shape[1], -1)

    time_major = jnp.swapaxes(flattened, 0, 1)
    hidden_state = tuple(
        module.init_state(
            plan.neuron,
            batch_size=batch_size,
            units=width,
            dtype=inputs.dtype,
        )
        for width in plan.hidden_sizes
    )
    output_state = module.init_state(
        plan.neuron,
        batch_size=batch_size,
        units=plan.num_classes,
        dtype=inputs.dtype,
    )
    output_linear_state = module.init_state(
        plan.neuron,
        batch_size=batch_size,
        units=plan.num_classes,
        dtype=inputs.dtype,
    )

    fixed_hidden = model_state["neuron_fixed"]["hidden"]
    fixed_output = model_state["neuron_fixed"]["output"]

    def scan_body(carry: Any, input_t: Any) -> tuple[Any, Any]:
        states, output_actual_state, output_free_state = carry
        next_states: list[Any] = []
        current = input_t

        for index in range(len(plan.hidden_sizes)):
            layer = params["hidden"][index]
            neuron_current = _apply_projection(current, layer)
            result = module.step(
                plan.neuron,
                layer["neuron"],
                fixed_hidden[index],
                states[index],
                neuron_current,
            )
            next_states.append(result.state)
            current = result.spike_output

        output_layer = params["output"]
        class_current = _apply_projection(current, output_layer)
        output_result = module.step(
            plan.neuron,
            output_layer["neuron"],
            fixed_output,
            output_actual_state,
            class_current,
        )
        next_output_free_state, reset_free_membrane = module.linear_step(
            plan.neuron,
            output_layer["neuron"],
            fixed_output,
            output_free_state,
            class_current,
        )
        next_carry = (
            tuple(next_states),
            output_result.state,
            next_output_free_state,
        )
        return next_carry, (output_result.spike_output, reset_free_membrane)

    _, (spikes_time, membrane_time) = jax.lax.scan(
        scan_body,
        (hidden_state, output_state, output_linear_state),
        time_major,
        unroll=plan.scan_unroll,
    )
    return ApplyResult(
        output=ModelOutput(
            spike_output=jnp.swapaxes(spikes_time, 0, 1),
            membrane_pre=jnp.swapaxes(membrane_time, 0, 1),
        ),
        model_state=model_state,
        traces={},
    )


def module_regularization_apply(
    params: Mapping[str, Any],
    model_state: Mapping[str, Any],
    inputs: Any,
    *,
    plan: ModelPlan,
) -> ApplyResult:

    return _module_trace_forward(
        params,
        model_state,
        inputs,
        plan=plan,
    )


def module_signal_apply(
    params: Mapping[str, Any],
    model_state: Mapping[str, Any],
    inputs: Any,
    *,
    plan: ModelPlan,
) -> ApplyResult:

    return _module_trace_forward(
        params,
        model_state,
        inputs,
        plan=plan,
    )


def _module_trace_forward(
    params: Mapping[str, Any],
    model_state: Mapping[str, Any],
    inputs: Any,
    *,
    plan: ModelPlan,
) -> ApplyResult:
    module = LIF
    batch_size = int(inputs.shape[0])
    flattened = inputs.reshape(batch_size, inputs.shape[1], -1)

    time_major = jnp.swapaxes(flattened, 0, 1)
    hidden_state = tuple(
        module.init_state(
            plan.neuron,
            batch_size=batch_size,
            units=width,
            dtype=inputs.dtype,
        )
        for width in plan.hidden_sizes
    )
    output_state = module.init_state(
        plan.neuron,
        batch_size=batch_size,
        units=plan.num_classes,
        dtype=inputs.dtype,
    )
    output_linear_state = module.init_state(
        plan.neuron,
        batch_size=batch_size,
        units=plan.num_classes,
        dtype=inputs.dtype,
    )
    fixed_hidden = model_state["neuron_fixed"]["hidden"]
    fixed_output = model_state["neuron_fixed"]["output"]

    def scan_body(carry: Any, input_t: Any) -> tuple[Any, Any]:
        states, output_actual_state, output_free_state = carry
        next_states: list[Any] = []
        current = input_t
        spike_layers: list[Any] = []

        for index in range(len(plan.hidden_sizes)):
            layer = params["hidden"][index]
            neuron_current = _apply_projection(current, layer)
            result = module.step(
                plan.neuron,
                layer["neuron"],
                fixed_hidden[index],
                states[index],
                neuron_current,
            )
            next_states.append(result.state)
            current = result.spike_output
            spike_layers.append(result.spike_output)

        output_layer = params["output"]
        class_current = _apply_projection(current, output_layer)
        output_result = module.step(
            plan.neuron,
            output_layer["neuron"],
            fixed_output,
            output_actual_state,
            class_current,
        )
        next_output_free_state, reset_free_membrane = module.linear_step(
            plan.neuron,
            output_layer["neuron"],
            fixed_output,
            output_free_state,
            class_current,
        )
        spike_layers.append(output_result.spike_output)
        next_carry = (
            tuple(next_states),
            output_result.state,
            next_output_free_state,
        )
        return next_carry, (
            output_result.spike_output,
            reset_free_membrane,
            tuple(spike_layers),
        )

    _, (spikes_time, membrane_time, traces_time) = jax.lax.scan(
        scan_body,
        (hidden_state, output_state, output_linear_state),
        time_major,
        unroll=plan.scan_unroll,
    )
    return ApplyResult(
        output=ModelOutput(
            spike_output=jnp.swapaxes(spikes_time, 0, 1),
            membrane_pre=jnp.swapaxes(membrane_time, 0, 1),
        ),
        model_state=model_state,
        traces={
            "spike_output": tuple(jnp.swapaxes(value, 0, 1) for value in traces_time)
        },
    )


__all__ = [
    "module_regularization_apply",
    "module_signal_apply",
    "module_eval_apply",
    "module_train_apply",
    "module_init",
]
