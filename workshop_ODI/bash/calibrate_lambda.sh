#!/usr/bin/env bash
set -euo pipefail

LAUNCHER_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(dirname "$(dirname "${LAUNCHER_PATH}")")"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=""
cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m workshop_jax.analysis.calibration_campaign "$@"
