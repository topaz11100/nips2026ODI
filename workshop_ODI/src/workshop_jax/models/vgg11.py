from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from workshop_jax.models.dataset_topology import (
    Vgg11Topology,
    resolve_vgg11_topology,
)
from workshop_jax.models.initialization_rng import InitializationKeyspace
from workshop_jax.models.spatial import (
    average_pool2d_time,
    batch_norm_frozen_time,
    batch_norm_train_time,
    conv2d_time,
    init_batch_norm_state,
    init_kaiming_normal_fan_out,
    init_normal,
    linear_time,
    run_output_neuron_site,
    run_output_neuron_site_signal,
    run_spatial_neuron_site,
    run_spatial_neuron_site_signal,
)
from workshop_jax.models.specification import ApplyResult, ModelPlan
from workshop_jax.training.initialization import initialize_neuron_layer


def _topology(plan: ModelPlan) -> Vgg11Topology:
    topology = resolve_vgg11_topology(
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
    input_channels = int(plan.canonical_input_shape[-1])
    for index, output_channels in enumerate(topology.conv_channels):
        site_path = ("conv", index + 1)
        neuron_params, neuron_fixed = initialize_neuron_layer(
            plan.neuron,
            units=output_channels,
        )
        hidden_params.append(
            {
                "weight": init_kaiming_normal_fan_out(
                    keyspace.topology_key((*site_path, "weight")),
                    (3, 3, input_channels, output_channels),
                ),
                "bn_scale": jnp.ones((output_channels,), dtype=jnp.float32),
                "bn_beta": jnp.zeros((output_channels,), dtype=jnp.float32),
                "neuron": neuron_params,
            }
        )
        hidden_fixed.append(neuron_fixed)
        normalization_state.append(init_batch_norm_state(output_channels))
        input_channels = output_channels
    classifier_site_path = ("classifier", 1)
    classifier_neuron_params, classifier_neuron_fixed = initialize_neuron_layer(
        plan.neuron,
        units=topology.classifier_width,
    )
    hidden_params.append(
        {
            "weight": init_normal(
                keyspace.topology_key((*classifier_site_path, "weight")),
                (topology.conv_channels[-1], topology.classifier_width),
                std=0.01,
            ),
            "bn_scale": jnp.ones((topology.classifier_width,), dtype=jnp.float32),
            "bn_beta": jnp.zeros((topology.classifier_width,), dtype=jnp.float32),
            "neuron": classifier_neuron_params,
        }
    )
    hidden_fixed.append(classifier_neuron_fixed)
    normalization_state.append(init_batch_norm_state(topology.classifier_width))
    output_site_path = ("output",)
    output_neuron_params, output_neuron_fixed = initialize_neuron_layer(
        plan.neuron,
        units=plan.num_classes,
    )
    output = {
        "weight": init_normal(
            keyspace.topology_key((*output_site_path, "weight")),
            (topology.classifier_width, plan.num_classes),
            std=0.01,
        ),
        "bias": jnp.zeros((plan.num_classes,), dtype=jnp.float32),
        "neuron": output_neuron_params,
    }
    params = {"hidden": tuple(hidden_params), "output": output}
    model_state = {
        "neuron_fixed": {
            "hidden": tuple(hidden_fixed),
            "output": output_neuron_fixed,
        },
        "normalization": {"hidden": tuple(normalization_state)},
    }
    return params, model_state


def _single_pass_convolution_trunk_train(
    inputs: Any,
    layers: Any,
    hidden_fixed: Any,
    normalization_state: Any,
    *,
    plan: ModelPlan,
    topology: Vgg11Topology,
) -> tuple[Any, tuple[dict[str, Any], ...]]:

    x = inputs
    next_norm: list[dict[str, Any]] = []
    pool_schedule_by_layer = _pool_schedule_by_layer(topology)
    for index, output_channels in enumerate(topology.conv_channels):
        layer = layers[index]
        current = conv2d_time(
            x,
            layer["weight"],
            None,
            stride=1,
            padding=1,
        )
        current, state = batch_norm_train_time(
            current,
            layer["bn_scale"],
            layer["bn_beta"],
            normalization_state[index],
        )
        next_norm.append(state)
        x = run_spatial_neuron_site(
            current,
            layer["neuron"],
            hidden_fixed[index],
            plan=plan,
        )

        pool_schedule = pool_schedule_by_layer.get(index)
        if pool_schedule is not None:
            pool_kernel, pool_stride = pool_schedule
            x = average_pool2d_time(
                x,
                kernel=pool_kernel,
                stride=pool_stride,
                padding=0,
            )

    return x, tuple(next_norm)


def _single_pass_convolution_trunk_frozen(
    inputs: Any,
    layers: Any,
    hidden_fixed: Any,
    normalization_state: Any,
    *,
    plan: ModelPlan,
    topology: Vgg11Topology,
) -> Any:

    x = inputs
    pool_schedule_by_layer = _pool_schedule_by_layer(topology)
    for index, output_channels in enumerate(topology.conv_channels):
        layer = layers[index]
        current = conv2d_time(
            x,
            layer["weight"],
            None,
            stride=1,
            padding=1,
        )
        current = batch_norm_frozen_time(
            current,
            layer["bn_scale"],
            layer["bn_beta"],
            normalization_state[index],
        )
        x = run_spatial_neuron_site(
            current,
            layer["neuron"],
            hidden_fixed[index],
            plan=plan,
        )

        pool_schedule = pool_schedule_by_layer.get(index)
        if pool_schedule is not None:
            pool_kernel, pool_stride = pool_schedule
            x = average_pool2d_time(
                x,
                kernel=pool_kernel,
                stride=pool_stride,
                padding=0,
            )

    return x


def _pool_schedule_by_layer(
    topology: Vgg11Topology,
) -> dict[int, tuple[int, int]]:
    return dict(
        zip(
            topology.pool_after,
            zip(topology.pool_kernels, topology.pool_strides, strict=True),
            strict=True,
        )
    )


def _flatten_classifier_input(inputs: Any, topology: Vgg11Topology) -> Any:
    flattened = inputs.reshape((inputs.shape[0], inputs.shape[1], -1))
    return flattened


def module_train_apply(
    params: Mapping[str, Any],
    model_state: Mapping[str, Any],
    inputs: Any,
    *,
    plan: ModelPlan,
) -> ApplyResult:

    topology = _topology(plan)
    x = jnp.asarray(inputs)
    hidden_fixed = model_state["neuron_fixed"]["hidden"]
    old_norm = model_state["normalization"]["hidden"]
    x, conv_norm = _single_pass_convolution_trunk_train(
        x,
        params["hidden"][: len(topology.conv_channels)],
        hidden_fixed[: len(topology.conv_channels)],
        old_norm[: len(topology.conv_channels)],
        plan=plan,
        topology=topology,
    )
    x = _flatten_classifier_input(x, topology)
    classifier_site_index = len(topology.conv_channels)
    classifier_layer = params["hidden"][classifier_site_index]
    classifier_current = linear_time(
        x,
        classifier_layer["weight"],
        None,
    )
    classifier_current, classifier_norm = batch_norm_train_time(
        classifier_current,
        classifier_layer["bn_scale"],
        classifier_layer["bn_beta"],
        old_norm[classifier_site_index],
    )
    x = run_spatial_neuron_site(
        classifier_current,
        classifier_layer["neuron"],
        hidden_fixed[classifier_site_index],
        plan=plan,
    )

    output_layer = params["output"]
    class_current = linear_time(
        x,
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
    next_model_state["normalization"] = {"hidden": (*conv_norm, classifier_norm)}
    return ApplyResult(output=output, model_state=next_model_state, traces={})


def module_eval_apply(
    params: Mapping[str, Any],
    model_state: Mapping[str, Any],
    inputs: Any,
    *,
    plan: ModelPlan,
) -> ApplyResult:

    topology = _topology(plan)
    x = jnp.asarray(inputs)
    hidden_fixed = model_state["neuron_fixed"]["hidden"]
    old_norm = model_state["normalization"]["hidden"]
    x = _single_pass_convolution_trunk_frozen(
        x,
        params["hidden"][: len(topology.conv_channels)],
        hidden_fixed[: len(topology.conv_channels)],
        old_norm[: len(topology.conv_channels)],
        plan=plan,
        topology=topology,
    )
    x = _flatten_classifier_input(x, topology)
    classifier_site_index = len(topology.conv_channels)
    classifier_layer = params["hidden"][classifier_site_index]
    classifier_current = linear_time(
        x,
        classifier_layer["weight"],
        None,
    )
    classifier_current = batch_norm_frozen_time(
        classifier_current,
        classifier_layer["bn_scale"],
        classifier_layer["bn_beta"],
        old_norm[classifier_site_index],
    )
    x = run_spatial_neuron_site(
        classifier_current,
        classifier_layer["neuron"],
        hidden_fixed[classifier_site_index],
        plan=plan,
    )

    output_layer = params["output"]
    class_current = linear_time(
        x,
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
    x = jnp.asarray(inputs)
    hidden_fixed = model_state["neuron_fixed"]["hidden"]
    old_norm = model_state["normalization"]["hidden"]
    next_norm: list[dict[str, Any]] = []
    site_traces: list[dict[str, Any]] = []
    pool_schedule_by_layer = _pool_schedule_by_layer(topology)
    for index, output_channels in enumerate(topology.conv_channels):
        layer = params["hidden"][index]
        current = conv2d_time(
            x,
            layer["weight"],
            None,
            stride=1,
            padding=1,
        )
        current, state = normalize(
            current,
            layer["bn_scale"],
            layer["bn_beta"],
            old_norm[index],
        )
        next_norm.append(state)
        x, traces = run_spatial_neuron_site_signal(
            current,
            layer["neuron"],
            hidden_fixed[index],
            plan=plan,
        )
        site_traces.append(traces)
        pool_schedule = pool_schedule_by_layer.get(index)
        if pool_schedule is not None:
            pool_kernel, pool_stride = pool_schedule
            pool_output = average_pool2d_time(
                x,
                kernel=pool_kernel,
                stride=pool_stride,
                padding=0,
            )
            x = pool_output

    x = _flatten_classifier_input(x, topology)
    classifier_site_index = len(topology.conv_channels)
    classifier_layer = params["hidden"][classifier_site_index]
    classifier_current = linear_time(
        x,
        classifier_layer["weight"],
        None,
    )
    classifier_current, classifier_norm = normalize(
        classifier_current,
        classifier_layer["bn_scale"],
        classifier_layer["bn_beta"],
        old_norm[classifier_site_index],
    )
    next_norm.append(classifier_norm)
    x, classifier_traces = run_spatial_neuron_site_signal(
        classifier_current,
        classifier_layer["neuron"],
        hidden_fixed[classifier_site_index],
        plan=plan,
    )
    site_traces.append(classifier_traces)

    output_layer = params["output"]
    class_current = linear_time(
        x,
        output_layer["weight"],
        output_layer["bias"],
    )
    output, output_traces = run_output_neuron_site_signal(
        class_current,
        output_layer["neuron"],
        model_state["neuron_fixed"]["output"],
        plan=plan,
    )
    site_traces.append(output_traces)
    next_model_state = dict(model_state)
    next_model_state["normalization"] = {"hidden": tuple(next_norm)}
    return ApplyResult(
        output=output,
        model_state=next_model_state,
        traces=_merge_traces(site_traces),
    )


def _module_signal_forward(
    params: Mapping[str, Any],
    model_state: Mapping[str, Any],
    inputs: Any,
    *,
    plan: ModelPlan,
) -> ApplyResult:
    topology = _topology(plan)
    x = jnp.asarray(inputs)
    hidden_fixed = model_state["neuron_fixed"]["hidden"]
    old_norm = model_state["normalization"]["hidden"]
    site_traces: list[dict[str, Any]] = []
    pool_schedule_by_layer = _pool_schedule_by_layer(topology)
    for index, output_channels in enumerate(topology.conv_channels):
        layer = params["hidden"][index]
        current = conv2d_time(
            x,
            layer["weight"],
            None,
            stride=1,
            padding=1,
        )
        current = batch_norm_frozen_time(
            current,
            layer["bn_scale"],
            layer["bn_beta"],
            old_norm[index],
        )
        x, traces = run_spatial_neuron_site_signal(
            current,
            layer["neuron"],
            hidden_fixed[index],
            plan=plan,
        )
        site_traces.append(traces)
        pool_schedule = pool_schedule_by_layer.get(index)
        if pool_schedule is not None:
            pool_kernel, pool_stride = pool_schedule
            pool_output = average_pool2d_time(
                x,
                kernel=pool_kernel,
                stride=pool_stride,
                padding=0,
            )
            x = pool_output

    x = _flatten_classifier_input(x, topology)
    classifier_site_index = len(topology.conv_channels)
    classifier_layer = params["hidden"][classifier_site_index]
    classifier_current = linear_time(
        x,
        classifier_layer["weight"],
        None,
    )
    classifier_current = batch_norm_frozen_time(
        classifier_current,
        classifier_layer["bn_scale"],
        classifier_layer["bn_beta"],
        old_norm[classifier_site_index],
    )
    x, classifier_traces = run_spatial_neuron_site_signal(
        classifier_current,
        classifier_layer["neuron"],
        hidden_fixed[classifier_site_index],
        plan=plan,
    )
    site_traces.append(classifier_traces)

    output_layer = params["output"]
    class_current = linear_time(
        x,
        output_layer["weight"],
        output_layer["bias"],
    )
    output, output_traces = run_output_neuron_site_signal(
        class_current,
        output_layer["neuron"],
        model_state["neuron_fixed"]["output"],
        plan=plan,
    )
    site_traces.append(output_traces)
    return ApplyResult(
        output=output,
        model_state=model_state,
        traces=_merge_traces(site_traces),
    )


def _merge_traces(
    site_traces: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Any, ...]]:
    return {"spike_output": tuple(site["spike_output"] for site in site_traces)}


__all__ = [
    "module_regularization_apply",
    "module_signal_apply",
    "module_eval_apply",
    "module_train_apply",
    "module_init",
]
