from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
from typing import Any

from workshop_jax.artifacts import load_yaml, write_csv
from workshop_jax.config import (
    CalibrationConfig,
    ConfigError,
    DATASETS,
    load_feature_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_calibration_config() -> CalibrationConfig:
    config = load_feature_config("lambda_calibration")
    if not isinstance(config, CalibrationConfig):
        raise ConfigError("expected lambda_calibration configuration")
    return config


def worker_command(config: CalibrationConfig, run: Path, dataset: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "workshop_jax.analysis.calibrate_lambda",
        "--artifact-root",
        str(config.artifact_root),
        "--batch-size",
        str(config.batch_size[dataset]),
        "--dataset",
        dataset,
        "--ratio",
        str(config.target_ratio),
        "--output",
        str(run / f"{dataset}.json"),
    ]


def worker_environment(
    config: CalibrationConfig, gpu: int, run: Path
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "JAX_ENABLE_X64": "true",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_CLIENT_MEM_FRACTION": str(config.memory_fraction),
            "XLA_FLAGS": f"--xla_gpu_force_compilation_parallelism={config.compile_threads}",
            "PYTHONPATH": str(PROJECT_ROOT / "src")
            + os.pathsep
            + environment.get("PYTHONPATH", ""),
            "JAX_COMPILATION_CACHE_DIR": str(run / "cache" / f"gpu_{gpu}"),
        }
    )
    return environment


def write_coefficients(
    output: Path, results: dict[str, Any], target_ratio: float
) -> Path:
    if set(results) != set(DATASETS):
        raise ValueError("lambda calibration requires all six completed datasets")
    rows = []
    for dataset in DATASETS:
        result = results[dataset]
        coefficient = float(result["lambda"])
        if (
            result["dataset"] != dataset
            or not math.isfinite(coefficient)
            or coefficient <= 0
        ):
            raise ValueError(f"invalid calibration coefficient: {dataset}")
        if result["target_ratio"] != target_ratio:
            raise ValueError(f"unexpected calibration ratio: {dataset}")
        rows.append({"dataset": dataset, "lambda": coefficient})
    write_csv(output, ("dataset", "lambda"), rows)
    return output


def _run_workers(config: CalibrationConfig, run: Path) -> dict[str, Any]:
    pending: Queue[str] = Queue()
    for dataset in (
        "cifar10-dvs",
        "cifar-100_16",
        "cifar-10_16",
        "dvs128-gesture",
        "s-mnist",
        "shd",
    ):
        pending.put(dataset)
    stop = threading.Event()
    processes: dict[int, Any] = {}
    results: dict[str, Any] = {}
    lock = threading.Lock()

    def terminate() -> None:
        stop.set()
        with lock:
            for process in processes.values():
                if process.poll() is None:
                    process.terminate()

    def worker(gpu: int) -> None:
        while not stop.is_set():
            try:
                dataset = pending.get_nowait()
            except Empty:
                return
            print(f"GPU {gpu}: START {dataset}", flush=True)
            log_path = run / f"{dataset}.log"
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    worker_command(config, run, dataset),
                    cwd=PROJECT_ROOT,
                    env=worker_environment(config, gpu, run),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                with lock:
                    processes[gpu] = process
                    if stop.is_set():
                        process.terminate()
                while True:
                    try:
                        code = process.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        if stop.is_set():
                            process.terminate()
                        lines = log_path.read_text(errors="replace").splitlines()
                        if lines:
                            print(f"GPU {gpu} {dataset}: {lines[-1]}", flush=True)
                with lock:
                    processes.pop(gpu, None)
                if code:
                    terminate()
                    details = log_path.read_text(errors="replace")[-8000:]
                    raise RuntimeError(f"{dataset} exited {code}\n{details}")
            result = json.loads((run / f"{dataset}.json").read_text())
            with lock:
                results[dataset] = result
            print(f"GPU {gpu}: COMPLETE {dataset}", flush=True)

    def handle_stop(signum: int, frame: Any) -> None:
        terminate()
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, handle_stop)
    try:
        with ThreadPoolExecutor(max_workers=len(config.gpu_ids)) as executor:
            futures = [executor.submit(worker, gpu) for gpu in config.gpu_ids]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                terminate()
                raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    return results


def run_campaign(config: CalibrationConfig) -> Path:
    load_yaml(config.artifact_root / "run_manifest.yaml")
    with TemporaryDirectory(prefix="workshop_odi_lambda_") as temporary:
        results = _run_workers(config, Path(temporary))
    return write_coefficients(config.output_csv, results, config.target_ratio)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate one SMR lambda per dataset on its prepared train probe."
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_calibration_config()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "feature": config.feature,
                    "gpu_ids": config.gpu_ids,
                    "target_ratio": config.target_ratio,
                    "batch_size": dict(config.batch_size),
                    "output_csv": str(config.output_csv),
                    "jobs": [
                        worker_command(config, Path("TEMPORARY_DIRECTORY"), dataset)
                        for dataset in DATASETS
                    ],
                },
                indent=2,
            )
        )
    else:
        print(run_campaign(config))


if __name__ == "__main__":
    main()
