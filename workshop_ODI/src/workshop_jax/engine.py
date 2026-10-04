from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
from typing import Sequence

from workshop_jax.artifacts import (
    archive_feature_stage,
    initialize_run_root,
    reusable_run_root,
    validate_dataset_signal_stage,
    validate_prepared_data_stage,
)
from workshop_jax.config import (
    AnalysisConfig,
    CalibrationConfig,
    ConfigError,
    DataPrepConfig,
    DATASETS,
    FEATURES,
    TrainConfig,
    load_feature_config,
    load_run_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _initialize(run_config_path: Path) -> None:
    config = load_run_config(run_config_path, PROJECT_ROOT)
    initialize_run_root(config.root, seed=config.seed, features=FEATURES)
    print(config.root)
    print(config.seed)


def _validate_train_signal_inputs(root: Path, seed: int) -> None:
    data_config = load_feature_config("data_prep")
    signal_config = load_feature_config("dataset_signal_analysis")
    if not isinstance(data_config, DataPrepConfig) or not isinstance(
        signal_config, AnalysisConfig
    ):
        raise ConfigError("invalid train_signal prerequisite config")
    validate_prepared_data_stage(
        root,
        datasets=data_config.datasets,
        seed=seed,
        train_probe_size=data_config.train_probe_size,
        test_probe_size=data_config.test_probe_size,
    )
    validate_dataset_signal_stage(
        root,
        datasets=signal_config.datasets,
        batch_size=signal_config.batch_size,
    )


def _reuse(
    run_config_path: Path,
    reset_feature: str | Sequence[str] | None,
) -> None:
    config = load_run_config(run_config_path, PROJECT_ROOT)
    root, seed = reusable_run_root(
        config.root,
        required_feature="data_prep",
        expected_datasets=DATASETS,
    )
    if reset_feature is None:
        reset_features: tuple[str, ...] = ()
    elif isinstance(reset_feature, str):
        reset_features = (reset_feature,)
    else:
        reset_features = tuple(reset_feature)
    allowed = {
        "train_regularization_off",
        "train_regularization_on",
        "signal_analysis_regularization_off",
    }
    unsupported = [feature for feature in reset_features if feature not in allowed]
    if unsupported:
        raise ConfigError(f"unsupported reusable feature reset: {unsupported[0]}")
    if len(set(reset_features)) != len(reset_features):
        raise ConfigError("reusable feature resets must be unique")
    if reset_features:
        _validate_train_signal_inputs(root, seed)
    for feature in reset_features:
        archive_feature_stage(root, feature)
    print(root)
    print(seed)


def _list_datasets(feature: str) -> None:
    config = load_feature_config(feature)
    for dataset in config.datasets:
        print(dataset)


def _configure_jax() -> None:
    visible = [
        token.strip()
        for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if token.strip()
    ]
    if len(visible) != 1:
        raise RuntimeError("each numerical child must see exactly one GPU")
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "default"
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
    import jax

    devices = tuple(jax.devices("gpu"))
    if len(devices) != 1:
        raise RuntimeError(f"expected one visible GPU, found {len(devices)}")


def _artifact_root(value: str) -> Path:
    root = Path(value).expanduser().resolve(strict=True)
    if not (root / "run_manifest.yaml").is_file():
        raise FileNotFoundError(root / "run_manifest.yaml")
    return root


def _run_feature(arguments: argparse.Namespace) -> Path | None:
    config = load_feature_config(arguments.feature)
    root = _artifact_root(arguments.artifact_root)
    seed = int(arguments.seed)
    phase = arguments.phase
    dataset = arguments.dataset
    if isinstance(config, CalibrationConfig):
        if phase != "execute" or dataset is not None:
            raise ConfigError(
                "lambda_calibration supports one feature-level execute phase"
            )
        from workshop_jax.analysis.calibration_campaign import run_campaign
        from workshop_jax.artifacts import load_yaml

        if seed != int(load_yaml(root / "run_manifest.yaml")["seed"]):
            raise ConfigError("lambda_calibration seed must match the prepared run")
        return run_campaign(replace(config, artifact_root=root))
    if isinstance(config, DataPrepConfig):
        if phase != "execute" or dataset is not None:
            raise ConfigError("data_prep supports only one feature-level execute phase")
        from workshop_jax.data.prepare import run_data_prep

        return run_data_prep(
            config,
            project_root=PROJECT_ROOT,
            artifact_root=root,
            seed=seed,
        )
    if (
        isinstance(config, AnalysisConfig)
        and config.feature == "dataset_signal_analysis"
    ):
        if phase == "finalize":
            from workshop_jax.analysis.dataset_reference import (
                finalize_dataset_references,
            )

            return finalize_dataset_references(config, artifact_root=root)
        if phase != "execute" or dataset is None:
            raise ConfigError("dataset_signal_analysis requires a dataset execute job")
        _configure_jax()
        from workshop_jax.analysis.dataset_reference import (
            run_dataset_reference_job,
        )

        return run_dataset_reference_job(
            config,
            artifact_root=root,
            dataset_name=dataset,
        )
    if isinstance(config, TrainConfig):
        if phase == "finalize":
            from workshop_jax.training.runner import finalize_training

            return finalize_training(config, artifact_root=root, seed=seed)
        if phase != "execute" or dataset is None:
            raise ConfigError("training requires a dataset execute job")
        _configure_jax()
        from workshop_jax.training.runner import run_training_job

        return run_training_job(
            config,
            artifact_root=root,
            dataset_name=dataset,
            seed=seed,
        )
    if not isinstance(config, AnalysisConfig):
        raise ConfigError(f"unsupported feature config: {config.feature}")
    if phase == "finalize":
        from workshop_jax.analysis.checkpoint_replay import (
            finalize_checkpoint_replay,
        )

        return finalize_checkpoint_replay(config, artifact_root=root, seed=seed)
    if phase != "execute" or dataset is None:
        raise ConfigError("signal analysis requires a dataset execute job")
    _configure_jax()
    from workshop_jax.analysis.checkpoint_replay import run_checkpoint_replay_job

    return run_checkpoint_replay_job(
        config,
        artifact_root=root,
        dataset_name=dataset,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m workshop_jax.engine")
    parser.add_argument(
        "--phase",
        required=True,
        choices=("initialize", "reuse", "list", "execute", "finalize"),
    )
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--feature")
    parser.add_argument("--artifact-root")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--reset-feature", action="append")
    arguments = parser.parse_args()
    try:
        if arguments.phase == "initialize":
            if arguments.run_config is None:
                raise ConfigError("initialize requires --run-config")
            _initialize(arguments.run_config)
            return
        if arguments.phase == "reuse":
            if arguments.run_config is None:
                raise ConfigError("reuse requires --run-config")
            _reuse(arguments.run_config, arguments.reset_feature)
            return
        if arguments.phase == "list":
            if arguments.feature is None:
                raise ConfigError("list requires --feature")
            _list_datasets(arguments.feature)
            return
        required = {
            "feature": arguments.feature,
            "artifact_root": arguments.artifact_root,
            "seed": arguments.seed,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ConfigError(f"missing engine arguments: {missing}")
        output = _run_feature(arguments)
        if output is not None:
            print(output)
    except (
        ConfigError,
        FileNotFoundError,
        FileExistsError,
        NotADirectoryError,
        ValueError,
    ) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()


__all__ = ["PROJECT_ROOT", "main"]
