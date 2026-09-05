from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

from workshop_jax.neurons.lif import NeuronPlan


@dataclass(frozen=True)
class ReadoutPlan:
    kind: str


@dataclass(frozen=True)
class ModelPlan:
    topology_id: str
    hidden_sizes: tuple[int, ...]
    canonical_input_shape: tuple[int, ...]
    input_dim: int
    num_classes: int
    neuron: NeuronPlan
    readout: ReadoutPlan
    dataset_selector: str
    scan_unroll: int = 392


class ModelOutput(NamedTuple):
    spike_output: Any
    membrane_pre: Any


class ApplyResult(NamedTuple):
    output: ModelOutput
    model_state: Any
    traces: Any
