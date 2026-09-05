from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp

from workshop_jax.models.dataset_topology import (
    ResNetBlockSpec,
    ResNetTopology,
    resolve_resnet_topology,
)
from workshop_jax.models.initialization_rng import InitializationKeyspace
from workshop_jax.models.spatial import (
    batch_norm_frozen_time,
    batch_norm_train_time,
    conv2d_time,
    init_batch_norm_state,
    init_kaiming_normal_fan_out,
    init_normal,
    linear_time,
    max_pool2d_time,
    run_output_neuron_site,
    run_output_neuron_site_signal,
    run_spatial_neuron_site,
    run_spatial_neuron_site_signal,
)
from workshop_jax.models.specification import ApplyResult, ModelPlan
from workshop_jax.training.initialization import initialize_neuron_layer


def _topology(plan: ModelPlan) -> ResNetTopology:
    topology = resolve_resnet_topology(
        plan.dataset_selector,
        tuple(int(value) for value in plan.canonical_input_shape),
    )
    return topology


def module_init(
    key: Any,
    plan: ModelPlan,
) -> tuple[dict[str, Any], dict[str, Any]]:
    topology = _topology(plan)
    keyspace = InitializationKeyspace(key, topology_id=plan.topology_id)
    hidden_params: list[dict[str, Any]] = []
    hidden_fixed: list[dict[str, Any]] = []
    normalization_state: list[dict[str, Any]] = []

    def append_site(
        site_id: str,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
    ) -> None:
        site_path = ("site", site_id)
        neuron_params, neuron_fixed = initialize_neuron_layer(
            plan.neuron,
            units=output_channels,
        )
        hidden_params.append(
            {
                "weight": init_kaiming_normal_fan_out(
                    keyspace.topology_key((*site_path, "weight")),
                    (kernel_size, kernel_size, input_channels, output_channels),
                ),
                "bn_scale": jnp.ones((output_channels,), dtype=jnp.float32),
                "bn_beta": jnp.zeros((output_channels,), dtype=jnp.float32),
                "neuron": neuron_params,
            }
        )
        hidden_fixed.append(neuron_fixed)
        normalization_state.append(init_batch_norm_state(output_channels))

    site_specs: dict[int, tuple[str, int, int, int]] = {}
    for spec in topology.blocks:
        if spec.adapter_site is not None:
            adapter_role = (
                "input_adapter" if spec.block_index == 0 else "channel_adapter"
            )
            site_specs[spec.adapter_site] = (
                f"{spec.block_id}_{adapter_role}",
                spec.input_channels,
                spec.output_channels,
                _adapter_kernel(spec),
            )
        site_specs[spec.main1_site] = (
            f"{spec.block_id}_main1",
            spec.output_channels,
            spec.output_channels,
            3,
        )
        site_specs[spec.main2_site] = (
            f"{spec.block_id}_main2",
            spec.output_channels,
            spec.output_channels,
            3,
        )
    for site_index in sorted(site_specs):
        append_site(*site_specs[site_index])
    output_site_path = ("output",)
    output_neuron, output_fixed = initialize_neuron_layer(
        plan.neuron,
        units=plan.num_classes,
    )
    output = {
        "weight": init_normal(
            keyspace.topology_key((*output_site_path, "weight")),
            (topology.output_width, plan.num_classes),
            std=0.01,
        ),
        "bias": jnp.zeros((plan.num_classes,), dtype=jnp.float32),
        "neuron": output_neuron,
    }
    params = {"hidden": tuple(hidden_params), "output": output}
    model_state = {
        "neuron_fixed": {
            "hidden": tuple(hidden_fixed),
            "output": output_fixed,
        },
        "normalization": {"hidden": tuple(normalization_state)},
    }
    return params, model_state


def _apply_pool(value: Any, spec: ResNetBlockSpec) -> Any:
    if spec.pool_kernel is None:
        return value
    if spec.pool_stride is None or spec.pool_padding is None:
        raise ValueError(f"incomplete pool specification for {spec.block_id}")
    return max_pool2d_time(
        value,
        kernel=spec.pool_kernel,
        stride=spec.pool_stride,
        padding=spec.pool_padding,
    )


def _adapter_kernel(spec: ResNetBlockSpec) -> int:
    if spec.adapter_kernel is None:
        raise ValueError(f"missing adapter kernel for {spec.block_id}")
    return spec.adapter_kernel


def _adapter_padding(spec: ResNetBlockSpec) -> int:
    if spec.adapter_padding is None:
        raise ValueError(f"missing adapter padding for {spec.block_id}")
    return spec.adapter_padding


def module_train_apply(
    params: Mapping[str, Any],
    model_state: Mapping[str, Any],
    inputs: Any,
    *,
    plan: ModelPlan,
) -> ApplyResult:

    topology = _topology(plan)
    value = jnp.asarray(inputs)
    fixed_layers = model_state["neuron_fixed"]["hidden"]
    old_norm = model_state["normalization"]["hidden"]
    next_norm: list[tuple[int, dict[str, Any]]] = []

    def apply_site(
        site_input: Any,
        *,
        site_index: int,
        padding: int,
    ) -> Any:
        layer = params["hidden"][site_index]
        current = conv2d_time(
            site_input,
            layer["weight"],
            None,
            stride=1,
            padding=padding,
        )
        current, state = batch_norm_train_time(
            current,
            layer["bn_scale"],
            layer["bn_beta"],
            old_norm[site_index],
        )
        next_norm.append((site_index, state))
        return run_spatial_neuron_site(
            current,
            layer["neuron"],
            fixed_layers[site_index],
            plan=plan,
        )

    for spec in topology.blocks:
        if spec.adapter_site is not None:
            value = apply_site(
                value,
                site_index=spec.adapter_site,
                padding=_adapter_padding(spec),
            )
        identity = value
        main = apply_site(value, site_index=spec.main1_site, padding=1)
        main = apply_site(main, site_index=spec.main2_site, padding=1)
        value = _apply_pool(main + identity, spec)

    pooled = _head_features(value, topology)
    output_layer = params["output"]
    class_current = linear_time(
        pooled,
        output_layer["weight"],
        output_layer["bias"],
    )
    output = run_output_neuron_site(
        class_current,
        output_layer["neuron"],
        model_state["neuron_fixed"]["output"],
        plan=plan,
    )
    next_model_state = dict(model_state)
    next_model_state["normalization"] = {
        "hidden": _ordered_norm_states(next_norm, len(old_norm))
    }
    return ApplyResult(output=output, model_state=next_model_state, traces={})


def module_eval_apply(
    params: Mapping[str, Any],
    model_state: Mapping[str, Any],
    inputs: Any,
    *,
    plan: ModelPlan,
) -> ApplyResult:

    topology = _topology(plan)
    value = jnp.asarray(inputs)
    fixed_layers = model_state["neuron_fixed"]["hidden"]
    old_norm = model_state["normalization"]["hidden"]

    def apply_site(
        site_input: Any,
        *,
        site_index: int,
        padding: int,
    ) -> Any:
        layer = params["hidden"][site_index]
        current = conv2d_time(
            site_input,
            layer["weight"],
            None,
            stride=1,
            padding=padding,
        )
        current = batch_norm_frozen_time(
            current,
            layer["bn_scale"],
            layer["bn_beta"],
            old_norm[site_index],
        )
        return run_spatial_neuron_site(
            current,
            layer["neuron"],
            fixed_layers[site_index],
            plan=plan,
        )

    for spec in topology.blocks:
        if spec.adapter_site is not None:
            value = apply_site(
                value,
                site_index=spec.adapter_site,
                padding=_adapter_padding(spec),
            )
        identity = value
        main = apply_site(value, site_index=spec.main1_site, padding=1)
        main = apply_site(main, site_index=spec.main2_site, padding=1)
        value = _apply_pool(main + identity, spec)

    pooled = _head_features(value, topology)
    output_layer = params["output"]
    class_current = linear_time(
        pooled,
        output_layer["weight"],
        output_layer["bias"],
    )
    output = run_output_neuron_site(
        class_current,
        output_layer["neuron"],
        model_state["neuron_fixed"]["output"],
        plan=plan,
    )
    return ApplyResult(output=output, model_state=model_state, traces={})


def module_regularization_apply(
    params: Mapping[str, Any],
    model_state: Mapping[str, Any],
    inputs: Any,
    *,
    plan: ModelPlan,
    normalization: Any = None,
) -> ApplyResult:
    return _module_regularization_forward(
        params,
        model_state,
        inputs,
        plan=plan,
        normalization=normalization,
    )


def module_signal_apply(
    params: Mapping[str, Any],
    model_state: Mapping[str, Any],
    inputs: Any,
    *,
    plan: ModelPlan,
) -> ApplyResult:
    return _module_signal_forward(
        params,
        model_state,
        inputs,
        plan=plan,
    )


def _module_regularization_forward(
    params: Mapping[str, Any],
    model_state: Mapping[str, Any],
    inputs: Any,
    *,
    plan: ModelPlan,
    normalization: Any = None,
) -> ApplyResult:

    normalize = batch_norm_train_time if normalization is None else normalization
    topology = _topology(plan)
    value = jnp.asarray(inputs)
    fixed_layers = model_state["neuron_fixed"]["hidden"]
    old_norm = model_state["normalization"]["hidden"]
    next_norm: list[tuple[int, dict[str, Any]]] = []
    site_traces: list[tuple[int, Mapping[str, Any]]] = []

    def apply_site(
        site_input: Any,
        *,
        site_index: int,
        padding: int,
    ) -> Any:
        layer = params["hidden"][site_index]
        current = conv2d_time(
            site_input,
            layer["weight"],
            None,
            stride=1,
            padding=padding,
        )
        current, state = normalize(
            current,
            layer["bn_scale"],
            layer["bn_beta"],
            old_norm[site_index],
        )
        next_norm.append((site_index, state))
        output, traces = run_spatial_neuron_site_signal(
            current,
            layer["neuron"],
            fixed_layers[site_index],
            plan=plan,
        )
        site_traces.append((site_index, traces))
        return output

    for spec in topology.blocks:
        if spec.adapter_site is not None:
            value = apply_site(
                value,
                site_index=spec.adapter_site,
                padding=_adapter_padding(spec),
            )
        identity = value
        main = apply_site(value, site_index=spec.main1_site, padding=1)
        main = apply_site(main, site_index=spec.main2_site, padding=1)
        value = _apply_pool(main + identity, spec)

    ordered_site_traces: list[Mapping[str, Any] | None] = [None] * len(
        plan.hidden_sizes
    )
    for index, traces in site_traces:
        ordered_site_traces[index] = traces

    pooled = _head_features(value, topology)
    output_layer = params["output"]
    class_current = linear_time(
        pooled,
        output_layer["weight"],
        output_layer["bias"],
    )
    output, output_traces = run_output_neuron_site_signal(
        class_current,
        output_layer["neuron"],
        model_state["neuron_fixed"]["output"],
        plan=plan,
    )
    next_model_state = dict(model_state)
    next_model_state["normalization"] = {
        "hidden": _ordered_norm_states(next_norm, len(old_norm))
    }
    return ApplyResult(
        output=output,
        model_state=next_model_state,
        traces=_merge_traces(
            [
                *(item for item in ordered_site_traces if item is not None),
                output_traces,
            ]
        ),
    )


def _module_signal_forward(
    params: Mapping[str, Any],
    model_state: Mapping[str, Any],
    inputs: Any,
    *,
    plan: ModelPlan,
) -> ApplyResult:

    topology = _topology(plan)
    value = jnp.asarray(inputs)
    fixed_layers = model_state["neuron_fixed"]["hidden"]
    old_norm = model_state["normalization"]["hidden"]
    site_traces: list[tuple[int, Mapping[str, Any]]] = []

    def apply_site(
        site_input: Any,
        *,
        site_index: int,
        padding: int,
    ) -> Any:
        layer = params["hidden"][site_index]
        current = conv2d_time(
            site_input,
            layer["weight"],
            None,
            stride=1,
            padding=padding,
        )
        current = batch_norm_frozen_time(
            current,
            layer["bn_scale"],
            layer["bn_beta"],
            old_norm[site_index],
        )
        output, traces = run_spatial_neuron_site_signal(
            current,
            layer["neuron"],
            fixed_layers[site_index],
            plan=plan,
        )
        site_traces.append((site_index, traces))
        return output

    for spec in topology.blocks:
        if spec.adapter_site is not None:
            value = apply_site(
                value,
                site_index=spec.adapter_site,
                padding=_adapter_padding(spec),
            )
        identity = value
        main = apply_site(value, site_index=spec.main1_site, padding=1)
        main = apply_site(main, site_index=spec.main2_site, padding=1)
        value = _apply_pool(main + identity, spec)

    ordered_site_traces: list[Mapping[str, Any] | None] = [None] * len(
        plan.hidden_sizes
    )
    for index, traces in site_traces:
        ordered_site_traces[index] = traces

    pooled = _head_features(value, topology)
    output_layer = params["output"]
    class_current = linear_time(
        pooled,
        output_layer["weight"],
        output_layer["bias"],
    )
    output, output_traces = run_output_neuron_site_signal(
        class_current,
        output_layer["neuron"],
        model_state["neuron_fixed"]["output"],
        plan=plan,
    )
    return ApplyResult(
        output=output,
        model_state=model_state,
        traces=_merge_traces(
            [
                *(item for item in ordered_site_traces if item is not None),
                output_traces,
            ]
        ),
    )


def _head_features(inputs: Any, topology: ResNetTopology) -> Any:
    return inputs.reshape((inputs.shape[0], inputs.shape[1], topology.output_width))


def _ordered_norm_states(
    indexed_states: list[tuple[int, dict[str, Any]]],
    count: int,
) -> tuple[dict[str, Any], ...]:
    ordered: list[dict[str, Any] | None] = [None] * count
    for index, state in indexed_states:
        ordered[index] = state
    return tuple(value for value in ordered if value is not None)


def _merge_traces(
    site_traces: list[Mapping[str, Any]],
) -> dict[str, tuple[Any, ...]]:
    return {"spike_output": tuple(site["spike_output"] for site in site_traces)}


__all__ = [
    "module_eval_apply",
    "module_init",
    "module_regularization_apply",
    "module_signal_apply",
    "module_train_apply",
]
