from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterator

import jax
import numpy as np

from workshop_jax.artifacts import feature_artifact_path, load_yaml
from workshop_jax.config import DATASETS
from workshop_jax.data.dataset import HostBatch, PreparedDataset
from workshop_jax.models import initialize
from workshop_jax.paper_spec import build_model_plan
from workshop_jax.training.calibration import calibrate_full_batch, _norm
from workshop_jax.training.runner import _device_batch


def cached_probe_batches(
    dataset: PreparedDataset, batch_size: int
) -> Callable[[], Iterator[HostBatch]]:
    indices = dataset.probe_indices("train")
    if (
        indices.ndim != 1
        or indices.size == 0
        or indices.dtype != np.dtype(np.int64)
        or np.any(indices < 0)
        or np.any(indices >= dataset.split_count("train"))
        or np.unique(indices).size != indices.size
    ):
        raise ValueError("invalid prepared train probe indices")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    cached = dataset.read_batch("train", indices, batch_size=indices.size)
    inputs, targets = cached.inputs, cached.targets

    def batches() -> Iterator[HostBatch]:
        for start in range(0, len(targets), batch_size):
            end = min(start + batch_size, len(targets))
            yield HostBatch(
                inputs[start:end], targets[start:end], np.ones(end - start, dtype=bool)
            )

    return batches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate SMR lambda at fixed initial parameters on the prepared train probe."
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.ratio) or args.ratio <= 0:
        parser.error("ratio must be positive and finite")
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if not jax.config.x64_enabled:
        raise RuntimeError("JAX_ENABLE_X64=true is required")
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("one visible GPU is required")
    root = args.artifact_root.resolve(strict=True)
    seed = int(load_yaml(root / "run_manifest.yaml")["seed"])
    dataset = PreparedDataset(feature_artifact_path(root, "data_prep") / args.dataset)
    plan = build_model_plan(dataset)
    plan = replace(plan, scan_unroll=16 if plan.topology_id == "dense_mlp_bias" else 1)
    params, model_state = initialize(jax.random.key(seed), plan)
    count = len(dataset.probe_indices("train"))
    batch_size = args.batch_size
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "phase": "cache_prepared_train_probe",
                "sample_count": count,
                "batch_size": batch_size,
            }
        ),
        flush=True,
    )
    host_batches = cached_probe_batches(dataset, batch_size)

    def batches() -> Iterator[Any]:
        for batch in host_batches():
            yield _device_batch(batch)

    def progress(values: dict[str, Any]) -> None:
        print(json.dumps({"dataset": args.dataset, **values}), flush=True)

    start = time.monotonic()
    result = calibrate_full_batch(
        params,
        model_state,
        plan=plan,
        batches=batches,
        sample_count=count,
        progress=progress,
    )
    coefficient = result.coefficient(args.ratio)
    payload = {
        "dataset": args.dataset,
        "lambda": coefficient,
        "target_ratio": args.ratio,
    }
    output.write_text(json.dumps(payload) + "\n")
    print(
        "COMPLETE "
        + json.dumps(
            {
                **payload,
                "sample_count": count,
                "task_gradient_norm": _norm(result.task_gradient["hidden"]),
                "smr_gradient_norm": _norm(result.smr_gradient["hidden"]),
                "task_loss": result.task_loss,
                "distance": result.distance,
                "normalization_sites": len(result.normalization_statistics),
                "elapsed_seconds": time.monotonic() - start,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
