#!/usr/bin/env bash
set -euo pipefail

LAUNCHER_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
source "$(dirname "${LAUNCHER_PATH}")/paper_data.sh"

paper_reg_main() {
  while (( $# )); do
    case "$1" in
      --config)
        if (( $# < 2 )); then
          printf '%s\n' '--config requires a YAML path' >&2
          return 2
        fi
        TRAIN_ON_CONFIG="$2"
        shift 2
        ;;
      --help|-h)
        printf 'Usage: %s [--config TRAINING_YAML]\n' "${LAUNCHER_PATH}"
        return 0
        ;;
      *)
        printf 'unknown argument: %s\n' "$1" >&2
        return 2
        ;;
    esac
  done
  local snapshot selection
  snapshot="$(mktemp)"
  if ! cp -- "${TRAIN_ON_CONFIG}" "${snapshot}" || ! selection="$(
    _run_cpu --phase list --feature train_regularization_on --config "${snapshot}"
  )"; then
    rm -f -- "${snapshot}"
    return 1
  fi
  if ! _initialize_context \
    reg \
    --phase reuse \
    --run-config "${RUN_CONFIG}" \
    --reset-feature train_regularization_on; then
    rm -f -- "${snapshot}"
    return 1
  fi
  TRAIN_ON_CONFIG="${LOG_DIR}/train_regularization_on.yaml"
  mv -- "${snapshot}" "${TRAIN_ON_CONFIG}"
  printf '%s\n' "${selection}" >"${LOG_DIR}/list_train_regularization_on.log"
  mapfile -t STAGE_DATASETS <<< "${selection}"
  _run_pool \
    train_regularization_on \
    "${TRAIN_ON_CONFIG}" \
    "${STAGE_DATASETS[@]}"
  _run_cpu_logged \
    finalize_train_regularization_on \
    --phase finalize \
    --feature train_regularization_on \
    --config "${TRAIN_ON_CONFIG}" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --seed "${SEED}"
  _print_result
}

paper_reg_main "$@"
