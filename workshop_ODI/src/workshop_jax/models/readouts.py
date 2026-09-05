from __future__ import annotations

from typing import Any

from workshop_jax.models.specification import ModelOutput, ReadoutPlan


def apply_readout(output: ModelOutput, plan: ReadoutPlan) -> Any:
    import jax.numpy as jnp

    if plan.kind == "max_rate":
        return jnp.mean(output.spike_output, axis=1)
    if plan.kind == "max_fire":
        return jnp.sum(output.spike_output, axis=1)
    if plan.kind == "final_mem":
        return output.membrane_pre[:, -1, :]
    if plan.kind == "temporal_mem":
        return jnp.mean(output.membrane_pre, axis=1)
    raise ValueError(f"unsupported paper readout: {plan.kind}")


def task_loss_elements(
    logits: Any,
    targets: Any,
    plan: ReadoutPlan,
    num_classes: int,
) -> Any:
    import jax.numpy as jnp
    import optax

    if plan.kind == "max_rate":
        one_hot = jnp.eye(num_classes, dtype=logits.dtype)[targets]
        return jnp.mean(jnp.square(logits - one_hot), axis=-1)
    if plan.kind in {"max_fire", "final_mem", "temporal_mem"}:
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    raise ValueError(f"unsupported paper readout: {plan.kind}")


def predictions(logits: Any) -> Any:
    import jax.numpy as jnp

    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


__all__ = [
    "apply_readout",
    "predictions",
    "task_loss_elements",
]
