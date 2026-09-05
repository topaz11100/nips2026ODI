from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from workshop_jax.analysis.spectral import batch_model_distance
from workshop_jax.models import eval_apply, regularization_apply, train_apply
from workshop_jax.models.readouts import apply_readout, predictions, task_loss_elements


class DeviceBatch(NamedTuple):
    inputs: Any
    targets: Any
    real_sample_mask: Any


class TrainingState(NamedTuple):
    params: Any
    model_state: Any
    opt_state: Any


class MetricSums(NamedTuple):
    loss_sum: Any
    correct_sum: Any
    count: Any


@dataclass(frozen=True)
class CompiledSteps:
    optimizer: Any
    train_step: Any
    eval_step: Any


def zero_sums() -> MetricSums:
    return MetricSums(
        loss_sum=jnp.asarray(0.0, dtype=jnp.float32),
        correct_sum=jnp.asarray(0, dtype=jnp.int64),
        count=jnp.asarray(0, dtype=jnp.int64),
    )


def _metric_sums(logits: Any, batch: DeviceBatch, plan: Any) -> MetricSums:
    losses = task_loss_elements(
        logits,
        batch.targets,
        plan.readout,
        plan.num_classes,
    )
    mask = jnp.asarray(batch.real_sample_mask, dtype=jnp.bool_)
    return MetricSums(
        loss_sum=jnp.sum(jnp.where(mask, losses, 0.0), dtype=jnp.float32),
        correct_sum=jnp.sum(
            jnp.where(mask, predictions(logits) == batch.targets, False),
            dtype=jnp.int64,
        ),
        count=jnp.sum(mask, dtype=jnp.int64),
    )


def _add_sums(left: MetricSums, right: MetricSums) -> MetricSums:
    return MetricSums(
        loss_sum=left.loss_sum + right.loss_sum,
        correct_sum=left.correct_sum + right.correct_sum,
        count=left.count + right.count,
    )


def build_steps(
    plan: Any,
    *,
    learning_rate: float,
    regularization_lambda: float,
) -> CompiledSteps:
    optimizer = optax.adam(
        learning_rate=float(learning_rate),
        b1=0.9,
        b2=0.999,
        eps=1.0e-8,
        eps_root=0.0,
    )
    expected_hidden_count = len(plan.hidden_sizes)

    def loss_with_aux(
        params: Any,
        model_state: Any,
        batch: DeviceBatch,
    ) -> tuple[Any, tuple[Any, MetricSums]]:
        if regularization_lambda > 0.0:
            applied = regularization_apply(
                params,
                model_state,
                batch.inputs,
                plan=plan,
            )
        else:
            applied = train_apply(
                params,
                model_state,
                batch.inputs,
                plan=plan,
            )
        logits = apply_readout(applied.output, plan.readout)
        metrics = _metric_sums(logits, batch, plan)
        denominator = jnp.maximum(metrics.count, jnp.asarray(1, dtype=jnp.int64))
        task_loss = metrics.loss_sum / denominator.astype(metrics.loss_sum.dtype)
        if regularization_lambda > 0.0:
            spike_traces = tuple(applied.traces["spike_output"])
            if len(spike_traces) != expected_hidden_count + 1:
                raise ValueError(
                    "regularization trace count does not match the model plan"
                )
            hidden_spikes = spike_traces[:-1]
            distance = batch_model_distance(
                batch.inputs,
                hidden_spikes,
                batch.real_sample_mask,
            )
            objective = (
                task_loss
                - jnp.asarray(regularization_lambda, dtype=jnp.float32) * distance
            )
        else:
            objective = task_loss
        return objective, (applied.model_state, metrics)

    def train_step(
        state: TrainingState,
        batch: DeviceBatch,
        totals: MetricSums,
    ) -> tuple[TrainingState, MetricSums]:
        (_, (candidate_model_state, metrics)), gradients = jax.value_and_grad(
            loss_with_aux,
            has_aux=True,
        )(state.params, state.model_state, batch)
        updates, opt_state = optimizer.update(
            gradients,
            state.opt_state,
            state.params,
        )
        next_state = TrainingState(
            params=optax.apply_updates(state.params, updates),
            model_state=candidate_model_state,
            opt_state=opt_state,
        )
        return next_state, _add_sums(totals, metrics)

    def eval_step(
        state: TrainingState,
        batch: DeviceBatch,
        totals: MetricSums,
    ) -> MetricSums:
        applied = eval_apply(
            state.params,
            state.model_state,
            batch.inputs,
            plan=plan,
        )
        logits = apply_readout(applied.output, plan.readout)
        return _add_sums(totals, _metric_sums(logits, batch, plan))

    return CompiledSteps(
        optimizer=optimizer,
        train_step=jax.jit(train_step, donate_argnums=(0, 2)),
        eval_step=jax.jit(eval_step, donate_argnums=(2,)),
    )


__all__ = [
    "CompiledSteps",
    "DeviceBatch",
    "MetricSums",
    "TrainingState",
    "build_steps",
    "zero_sums",
]
