from __future__ import annotations

from typing import Any


def _module(topology: str) -> Any:
    if topology == "dense_mlp_bias":
        from workshop_jax.models import dense

        return dense
    if topology == "vgg11":
        from workshop_jax.models import vgg11

        return vgg11
    if topology == "resnet":
        from workshop_jax.models import resnet

        return resnet
    raise ValueError(f"unsupported paper topology: {topology}")


def initialize(key: Any, plan: Any) -> tuple[Any, Any]:
    return _module(plan.topology_id).module_init(key, plan)


def train_apply(
    params: Any,
    model_state: Any,
    inputs: Any,
    *,
    plan: Any,
) -> Any:
    return _module(plan.topology_id).module_train_apply(
        params,
        model_state,
        inputs,
        plan=plan,
    )


def eval_apply(
    params: Any,
    model_state: Any,
    inputs: Any,
    *,
    plan: Any,
) -> Any:
    return _module(plan.topology_id).module_eval_apply(
        params,
        model_state,
        inputs,
        plan=plan,
    )


def regularization_apply(
    params: Any,
    model_state: Any,
    inputs: Any,
    *,
    plan: Any,
    normalization: Any = None,
) -> Any:
    options = {} if normalization is None else {"normalization": normalization}
    return _module(plan.topology_id).module_regularization_apply(
        params,
        model_state,
        inputs,
        plan=plan,
        **options,
    )


def signal_apply(
    params: Any,
    model_state: Any,
    inputs: Any,
    *,
    plan: Any,
) -> Any:
    return _module(plan.topology_id).module_signal_apply(
        params,
        model_state,
        inputs,
        plan=plan,
    )


__all__ = [
    "eval_apply",
    "initialize",
    "regularization_apply",
    "signal_apply",
    "train_apply",
]
