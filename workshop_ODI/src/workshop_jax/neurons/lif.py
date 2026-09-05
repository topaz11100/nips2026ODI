from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any, Mapping, NamedTuple


@dataclass(frozen=True)
class NeuronPlan:
    threshold: float = 1.0
    alpha: float = 0.5


class NeuronStepResult(NamedTuple):
    state: Any
    membrane_pre: Any
    membrane_post: Any
    spike_output: Any
    auxiliary: Any = None


@cache
def _surrogate() -> Any:
    import jax
    import jax.numpy as jnp

    @jax.custom_vjp
    def threshold(value: Any) -> Any:
        return (value >= 0).astype(value.dtype)

    def forward(value: Any) -> tuple[Any, Any]:
        return (value >= 0).astype(value.dtype), value

    def backward(value: Any, cotangent: Any) -> tuple[Any]:
        derivative = 0.1 / (1.0 + jnp.square(0.1 * jnp.pi * value))
        return (cotangent * derivative,)

    threshold.defvjp(forward, backward)
    return threshold


class LIFModule:
    def initialize_layer(
        self,
        plan: NeuronPlan,
        *,
        units: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        import jax.numpy as jnp

        fixed = {
            "threshold": jnp.full((units,), plan.threshold, dtype=jnp.float32),
            "alpha": jnp.full((units,), plan.alpha, dtype=jnp.float32),
        }
        return {}, fixed

    def init_state(
        self,
        plan: NeuronPlan,
        *,
        batch_size: int,
        units: int,
        dtype: Any,
    ) -> dict[str, Any]:
        import jax.numpy as jnp

        del plan
        return {"membrane_post_t_minus_1": jnp.zeros((batch_size, units), dtype=dtype)}

    def step(
        self,
        plan: NeuronPlan,
        params: Mapping[str, Any],
        fixed: Mapping[str, Any],
        state: Mapping[str, Any],
        input_current: Any,
    ) -> NeuronStepResult:
        del params
        previous = state["membrane_post_t_minus_1"]
        membrane_pre = fixed["alpha"] * previous + input_current
        spike = _surrogate()(membrane_pre - fixed["threshold"])
        membrane_post = membrane_pre - fixed["threshold"] * spike
        return NeuronStepResult(
            state={"membrane_post_t_minus_1": membrane_post},
            membrane_pre=membrane_pre,
            membrane_post=membrane_post,
            spike_output=spike,
        )

    def linear_step(
        self,
        plan: NeuronPlan,
        params: Mapping[str, Any],
        fixed: Mapping[str, Any],
        state: Mapping[str, Any],
        input_current: Any,
    ) -> tuple[dict[str, Any], Any]:
        del plan, params
        membrane = fixed["alpha"] * state["membrane_post_t_minus_1"] + input_current
        return {"membrane_post_t_minus_1": membrane}, membrane


LIF = LIFModule()


__all__ = [
    "LIF",
    "LIFModule",
    "NeuronPlan",
    "NeuronStepResult",
]
