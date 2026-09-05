from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping

from workshop_jax.models.dataset_topology import (
    resolve_resnet_topology,
    resolve_vgg11_topology,
)
from workshop_jax.models.specification import ModelPlan, ReadoutPlan
from workshop_jax.neurons.lif import NeuronPlan


@dataclass(frozen=True)
class PaperExperimentSpec:
    dataset: str
    topology: str
    hidden_sizes: tuple[int, ...]
    readout: str


PAPER_EXPERIMENTS: Mapping[str, PaperExperimentSpec] = {
    "s-mnist": PaperExperimentSpec(
        dataset="s-mnist",
        topology="dense_mlp_bias",
        hidden_sizes=(128, 128, 128, 128, 128),
        readout="max_rate",
    ),
    "shd": PaperExperimentSpec(
        dataset="shd",
        topology="dense_mlp_bias",
        hidden_sizes=(128, 128, 128),
        readout="max_rate",
    ),
    "cifar-10_16": PaperExperimentSpec(
        dataset="cifar-10_16",
        topology="vgg11",
        hidden_sizes=(),
        readout="final_mem",
    ),
    "dvs128-gesture": PaperExperimentSpec(
        dataset="dvs128-gesture",
        topology="vgg11",
        hidden_sizes=(),
        readout="final_mem",
    ),
    "cifar-100_16": PaperExperimentSpec(
        dataset="cifar-100_16",
        topology="resnet",
        hidden_sizes=(),
        readout="final_mem",
    ),
    "cifar10-dvs": PaperExperimentSpec(
        dataset="cifar10-dvs",
        topology="resnet",
        hidden_sizes=(),
        readout="final_mem",
    ),
}


def experiment_spec(dataset: str) -> PaperExperimentSpec:
    try:
        return PAPER_EXPERIMENTS[str(dataset)]
    except KeyError as error:
        raise ValueError(f"unsupported paper dataset: {dataset}") from error


def build_model_plan(dataset: Any) -> ModelPlan:
    spec = experiment_spec(dataset.dataset_id)
    canonical_shape = tuple(int(value) for value in dataset.canonical_sample_shape)
    if spec.topology == "vgg11":
        hidden_sizes = resolve_vgg11_topology(
            spec.dataset, canonical_shape
        ).hidden_site_widths
    elif spec.topology == "resnet":
        hidden_sizes = resolve_resnet_topology(
            spec.dataset, canonical_shape
        ).hidden_site_widths
    else:
        hidden_sizes = spec.hidden_sizes
    neuron = NeuronPlan()
    readout = ReadoutPlan(kind=spec.readout)
    return ModelPlan(
        topology_id=spec.topology,
        hidden_sizes=tuple(int(value) for value in hidden_sizes),
        canonical_input_shape=canonical_shape,
        input_dim=int(prod(dataset.canonical_sample_shape[1:])),
        num_classes=int(dataset.num_classes),
        neuron=neuron,
        readout=readout,
        dataset_selector=spec.dataset,
    )


__all__ = [
    "PAPER_EXPERIMENTS",
    "PaperExperimentSpec",
    "build_model_plan",
    "experiment_spec",
]
