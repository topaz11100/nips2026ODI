#!/usr/bin/env bash
set -euo pipefail

LAUNCHER_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
source "$(dirname "${LAUNCHER_PATH}")/paper_data.sh"

paper_reg_main() {
  while (( $# )); do
    case "$1" in
      --help|-h)
        printf 'Usage: %s\n' "${LAUNCHER_PATH}"
        return 0
        ;;
      *)
        printf 'unknown argument: %s\n' "$1" >&2
        return 2
        ;;
    esac
  done
  local selection
  if ! selection="$(
    _run_cpu --phase list --feature train_regularization_on
  )"; then
    return 1
  fi
  _initialize_context \
    reg \
    --phase reuse \
    --run-config "${RUN_CONFIG}" \
    --reset-feature train_regularization_on
  printf '%s\n' "${selection}" >"${LOG_DIR}/list_train_regularization_on.log"
  mapfile -t STAGE_DATASETS <<< "${selection}"
  _run_pool \
    train_regularization_on \
    "${STAGE_DATASETS[@]}"
  _run_cpu_logged \
    finalize_train_regularization_on \
    --phase finalize \
    --feature train_regularization_on \
    --artifact-root "${ARTIFACT_ROOT}" \
    --seed "${SEED}"
  _print_result
}

paper_reg_main "$@"
