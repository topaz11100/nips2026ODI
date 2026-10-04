#!/usr/bin/env bash
set -euo pipefail

LAUNCHER_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
source "$(dirname "${LAUNCHER_PATH}")/paper_data.sh"

paper_train_signal_main() {
  _initialize_context \
    train_signal \
    --phase reuse \
    --run-config "${RUN_CONFIG}" \
    --reset-feature train_regularization_off \
    --reset-feature signal_analysis_regularization_off
  _load_datasets train_regularization_off
  _run_pool \
    train_regularization_off \
    "${STAGE_DATASETS[@]}"
  _run_cpu_logged \
    finalize_train_regularization_off \
    --phase finalize \
    --feature train_regularization_off \
    --artifact-root "${ARTIFACT_ROOT}" \
    --seed "${SEED}"
  _load_datasets signal_analysis_regularization_off
  _run_pool \
    signal_analysis_regularization_off \
    "${STAGE_DATASETS[@]}"
  _run_cpu_logged \
    finalize_signal_analysis_regularization_off \
    --phase finalize \
    --feature signal_analysis_regularization_off \
    --artifact-root "${ARTIFACT_ROOT}" \
    --seed "${SEED}"
  _print_result
}

paper_train_signal_main "$@"
