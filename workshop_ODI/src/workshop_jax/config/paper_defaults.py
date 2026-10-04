"""Fixed feature settings formerly stored in the six paper feature YAMLs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


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

PROBE_SIZE = {
    "s-mnist": 160,
    "shd": 320,
    "cifar-10_16": 160,
    "dvs128-gesture": 176,
    "cifar-100_16": 1600,
    "cifar10-dvs": 160,
}
BATCH_SIZE = {
    "s-mnist": 256,
    "shd": 128,
    "cifar-10_16": 256,
    "dvs128-gesture": 16,
    "cifar-100_16": 256,
    "cifar10-dvs": 16,
}

# When recalibrating, explicitly copy the chosen CSV values here before SMR
# training. Training never replaces these constants with values from a CSV.
REGULARIZATION_LAMBDA = {
    "s-mnist": 0.00015369710743103033,
    "shd": 0.000039783069496538944,
    "cifar-10_16": 1.6394609052715678,
    "dvs128-gesture": 1.500931727993789,
    "cifar-100_16": 0.2460894605100541,
    "cifar10-dvs": 0.23780889087008814,
}

CALIBRATION_ARTIFACT_ROOT = "paper_artifacts"
CALIBRATION_OUTPUT_CSV = "paper_artifacts/lambda_calibration.csv"

FEATURE_DEFAULTS = {
    "data_prep": {
        "datasets": list(DATASETS),
        "train_probe_size": PROBE_SIZE,
        "test_probe_size": PROBE_SIZE,
    },
    "dataset_signal_analysis": {
        "datasets": list(DATASETS),
        "batch_size": BATCH_SIZE,
    },
    "train_regularization_off": {
        "datasets": list(DATASETS),
        "epochs": 50,
        "checkpoint_epochs": [1, 5, 10, 50],
        "learning_rate": 0.0025,
        "batch_size": BATCH_SIZE,
    },
    "train_regularization_on": {
        "datasets": list(DATASETS),
        "epochs": 50,
        "learning_rate": 0.0025,
        "lambda": REGULARIZATION_LAMBDA,
        "batch_size": BATCH_SIZE,
    },
    "signal_analysis_regularization_off": {
        "datasets": list(DATASETS),
        "batch_size": BATCH_SIZE,
    },
    "lambda_calibration": {
        "artifact_root": CALIBRATION_ARTIFACT_ROOT,
        "output_csv": CALIBRATION_OUTPUT_CSV,
        "batch_size": BATCH_SIZE,
        "target_ratio": 0.1,
        "gpu_ids": [0, 1],
        "memory_fraction": 0.9,
        "compile_threads": 4,
    },
}


def feature_payload(feature: str) -> dict[str, Any]:
    """Return fresh settings so callers cannot mutate subsequent loads."""
    return deepcopy(FEATURE_DEFAULTS[feature])
