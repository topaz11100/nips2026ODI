from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


DATASETS = (
    "s-mnist",
    "shd",
    "cifar-10_16",
    "dvs128-gesture",
    "cifar-100_16",
    "cifar10-dvs",
)
FEATURES = (
    "data_prep",
    "dataset_signal_analysis",
    "train_regularization_off",
    "train_regularization_on",
    "signal_analysis_regularization_off",
    "lambda_calibration",
)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RunConfig:
    root: Path
    seed: int


@dataclass(frozen=True)
class DataPrepConfig:
    feature: str
    datasets: tuple[str, ...]
    train_probe_size: Mapping[str, int]
    test_probe_size: Mapping[str, int]


@dataclass(frozen=True)
class AnalysisConfig:
    feature: str
    datasets: tuple[str, ...]
    batch_size: Mapping[str, int]


@dataclass(frozen=True)
class TrainConfig:
    feature: str
    datasets: tuple[str, ...]
    epochs: int
    learning_rate: float
    batch_size: Mapping[str, int]
    regularization_lambda: Mapping[str, float]
    checkpoint_epochs: tuple[int, ...] = ()


@dataclass(frozen=True)
class CalibrationConfig:
    artifact_root: Path
    output_csv: Path
    batch_size: Mapping[str, int]
    target_ratio: float
    gpu_ids: tuple[int, ...]
    memory_fraction: float
    compile_threads: int
    feature: str = "lambda_calibration"
    datasets: tuple[str, ...] = DATASETS


FeatureConfig = DataPrepConfig | AnalysisConfig | TrainConfig | CalibrationConfig


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{location} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    actual = {str(key) for key in value}
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ConfigError(
            f"{location} has invalid keys; missing={missing}, unknown={unknown}"
        )


def _document(path: Path) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return _mapping(value, str(source))


def _datasets(value: Any, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{location} must be a nonempty list")
    result = tuple(str(item) for item in value)
    if result != DATASETS:
        raise ConfigError(f"{location} must match the fixed paper dataset inventory")
    return result


def _positive_map(
    value: Any,
    datasets: tuple[str, ...],
    location: str,
) -> dict[str, int]:
    raw = _mapping(value, location)
    _exact_keys(raw, set(datasets), location)
    if any(type(raw[dataset]) is not int for dataset in datasets):
        raise ConfigError(f"{location} values must be positive integers")
    result = {dataset: int(raw[dataset]) for dataset in datasets}
    if any(number <= 0 for number in result.values()):
        raise ConfigError(f"{location} values must be positive integers")
    return result


def _positive_float(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{location} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ConfigError(f"{location} must be a finite positive number")
    return result


def _positive_float_map(
    value: Any, datasets: tuple[str, ...], location: str
) -> dict[str, float]:
    raw = _mapping(value, location)
    _exact_keys(raw, set(datasets), location)
    return {
        dataset: _positive_float(raw[dataset], f"{location}.{dataset}")
        for dataset in datasets
    }


def _checkpoint_epochs(value: Any, epochs: int, location: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{location} must be a nonempty list")
    if any(type(epoch) is not int for epoch in value):
        raise ConfigError(f"{location} values must be integers")
    result = tuple(int(epoch) for epoch in value)
    if any(epoch <= 0 or epoch > epochs for epoch in result):
        raise ConfigError(f"{location} values must be in [1, epochs]")
    if result != tuple(sorted(set(result))):
        raise ConfigError(f"{location} values must be unique and increasing")
    return result


def load_run_config(path: Path, project_root: Path) -> RunConfig:
    document = _document(path)
    _exact_keys(document, {"paper"}, "run document")
    payload = _mapping(document["paper"], "paper")
    _exact_keys(payload, {"root", "seed"}, "paper")
    root_value = payload["root"]
    if not isinstance(root_value, str) or not root_value.strip():
        raise ConfigError("paper.root must be a nonempty path string")
    configured_root = Path(root_value).expanduser()
    root = (
        configured_root
        if configured_root.is_absolute()
        else Path(project_root).resolve(strict=True) / configured_root
    ).resolve(strict=False)
    if type(payload["seed"]) is not int:
        raise ConfigError("paper.seed must be an integer")
    seed = int(payload["seed"])
    if seed < 0 or seed > 0xFFFFFFFF:
        raise ConfigError("paper.seed must be in [0, 4294967295]")
    return RunConfig(root=root, seed=seed)


def load_feature_config(
    path: Path, expected_feature: str | None = None
) -> FeatureConfig:
    document = _document(path)
    if len(document) != 1:
        raise ConfigError("a feature document must contain exactly one root")
    feature = str(next(iter(document)))
    if feature not in FEATURES:
        raise ConfigError(f"unsupported feature root: {feature}")
    if expected_feature is not None and feature != expected_feature:
        raise ConfigError(f"expected {expected_feature}, found {feature}")
    payload = _mapping(document[feature], feature)
    if feature == "lambda_calibration":
        _exact_keys(
            payload,
            {
                "artifact_root",
                "output_csv",
                "batch_size",
                "target_ratio",
                "gpu_ids",
                "memory_fraction",
                "compile_threads",
            },
            feature,
        )
        ratio = _positive_float(payload["target_ratio"], f"{feature}.target_ratio")
        fraction = _positive_float(
            payload["memory_fraction"], f"{feature}.memory_fraction"
        )
        if fraction >= 1:
            raise ConfigError(
                "lambda_calibration.memory_fraction must be less than one"
            )
        gpus = payload["gpu_ids"]
        if (
            not isinstance(gpus, list)
            or not gpus
            or any(type(gpu) is not int or gpu < 0 for gpu in gpus)
            or len(set(gpus)) != len(gpus)
        ):
            raise ConfigError(
                "lambda_calibration.gpu_ids must be distinct nonnegative integers"
            )
        threads = payload["compile_threads"]
        if type(threads) is not int or threads <= 0:
            raise ConfigError(
                "lambda_calibration.compile_threads must be a positive integer"
            )
        project = Path(__file__).resolve().parents[3]

        def resolve(name: str) -> Path:
            raw = payload[name]
            if not isinstance(raw, str) or not raw.strip():
                raise ConfigError(f"{feature}.{name} must be a nonempty path string")
            value = Path(raw).expanduser()
            return (value if value.is_absolute() else project / value).resolve()

        output_csv = resolve("output_csv")
        if output_csv.suffix != ".csv":
            raise ConfigError("lambda_calibration.output_csv must end in .csv")
        return CalibrationConfig(
            resolve("artifact_root"),
            output_csv,
            _positive_map(payload["batch_size"], DATASETS, f"{feature}.batch_size"),
            ratio,
            tuple(gpus),
            fraction,
            threads,
        )
    if feature == "data_prep":
        _exact_keys(
            payload,
            {"datasets", "train_probe_size", "test_probe_size"},
            feature,
        )
        datasets = _datasets(
            payload["datasets"],
            f"{feature}.datasets",
        )
        return DataPrepConfig(
            feature=feature,
            datasets=datasets,
            train_probe_size=_positive_map(
                payload["train_probe_size"], datasets, f"{feature}.train_probe_size"
            ),
            test_probe_size=_positive_map(
                payload["test_probe_size"], datasets, f"{feature}.test_probe_size"
            ),
        )
    if feature in {
        "dataset_signal_analysis",
        "signal_analysis_regularization_off",
    }:
        _exact_keys(payload, {"datasets", "batch_size"}, feature)
        datasets = _datasets(payload["datasets"], f"{feature}.datasets")
        return AnalysisConfig(
            feature=feature,
            datasets=datasets,
            batch_size=_positive_map(
                payload["batch_size"], datasets, f"{feature}.batch_size"
            ),
        )
    expected = {"datasets", "epochs", "learning_rate", "batch_size"}
    if feature == "train_regularization_on":
        expected.add("lambda")
    else:
        expected.add("checkpoint_epochs")
    _exact_keys(payload, expected, feature)
    datasets = _datasets(payload["datasets"], f"{feature}.datasets")
    if type(payload["epochs"]) is not int:
        raise ConfigError(f"{feature}.epochs must be a positive integer")
    epochs = int(payload["epochs"])
    learning_rate = _positive_float(
        payload["learning_rate"], f"{feature}.learning_rate"
    )
    regularization_lambda = (
        _positive_float_map(payload["lambda"], datasets, f"{feature}.lambda")
        if feature == "train_regularization_on"
        else {dataset: 0.0 for dataset in datasets}
    )
    if epochs <= 0:
        raise ConfigError(f"{feature}.epochs must be a positive integer")
    checkpoint_epochs = (
        _checkpoint_epochs(
            payload["checkpoint_epochs"],
            epochs,
            f"{feature}.checkpoint_epochs",
        )
        if feature == "train_regularization_off"
        else ()
    )
    return TrainConfig(
        feature=feature,
        datasets=datasets,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=_positive_map(
            payload["batch_size"], datasets, f"{feature}.batch_size"
        ),
        regularization_lambda=regularization_lambda,
        checkpoint_epochs=checkpoint_epochs,
    )


__all__ = [
    "AnalysisConfig",
    "CalibrationConfig",
    "ConfigError",
    "DATASETS",
    "DataPrepConfig",
    "FEATURES",
    "FeatureConfig",
    "RunConfig",
    "TrainConfig",
    "load_feature_config",
    "load_run_config",
]
