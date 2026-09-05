from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from workshop_jax.neurons.lif import LIF, NeuronPlan


def kaiming_weight(key: Any, fan_in: int, fan_out: int) -> Any:
    scale = math.sqrt(2.0 / fan_in)
    return jax.random.normal(key, (fan_in, fan_out), dtype=jnp.float32) * jnp.asarray(
        scale, dtype=jnp.float32
    )


def initialize_neuron_layer(
    plan: NeuronPlan,
    *,
    units: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return LIF.initialize_layer(plan, units=units)


__all__ = [
    "initialize_neuron_layer",
    "kaiming_weight",
]
