#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
IO_WORKERS="${WORKSHOP_ODI_IO_WORKERS:-8}"
JAX_COMPILE_THREADS="${WORKSHOP_ODI_JAX_COMPILE_THREADS:-4}"
GPU_FREE_MIB="${WORKSHOP_ODI_GPU_FREE_MIB:-512}"
CPU_AFFINITY="${WORKSHOP_ODI_CPU_AFFINITY-0-15}"
CACHE_ROOT="${WORKSHOP_ODI_CACHE_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/workshop_odi}"
MEMORY_FRACTION="${WORKSHOP_ODI_MEMORY_FRACTION:-0.75}"
GPU_IDS_RAW="${WORKSHOP_ODI_GPU_IDS:-0 1}"
read -r -a GPU_IDS <<< "${GPU_IDS_RAW}"

RUN_CONFIG="${PROJECT_ROOT}/config/paper/run.yaml"

if (( ${#GPU_IDS[@]} == 0 )); then
  printf 'WORKSHOP_ODI_GPU_IDS must contain at least one GPU\n' >&2
  exit 2
fi

declare -A SEEN_GPUS=()
for gpu_id in "${GPU_IDS[@]}"; do
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]] || [[ -n "${SEEN_GPUS[${gpu_id}]:-}" ]]; then
    printf 'invalid WORKSHOP_ODI_GPU_IDS value: %s\n' "${GPU_IDS_RAW}" >&2
    exit 2
  fi
  SEEN_GPUS["${gpu_id}"]=1
done

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export WORKSHOP_ODI_GPU_IDS="${GPU_IDS[*]}"
export WORKSHOP_ODI_IO_WORKERS="${IO_WORKERS}"
export WORKSHOP_ODI_JAX_COMPILE_THREADS="${JAX_COMPILE_THREADS}"
export WORKSHOP_ODI_CPU_AFFINITY="${CPU_AFFINITY}"
export WORKSHOP_ODI_CACHE_ROOT="${CACHE_ROOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE="true"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEMORY_FRACTION}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="-1"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_force_compilation_parallelism=${JAX_COMPILE_THREADS}"

_engine() {
  "${PYTHON_BIN}" -m workshop_jax.engine "$@"
}

_run_cpu() {
  CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    _engine "$@"
}

_initialize_context() {
  local label="$1"
  shift
  local output
  local -a run_values=()
  if ! output="$(_run_cpu "$@")"; then
    return 1
  fi
  mapfile -t run_values <<< "${output}"
  if (( ${#run_values[@]} != 2 )); then
    printf 'run initialization returned an invalid result\n' >&2
    return 2
  fi
  ARTIFACT_ROOT="${run_values[0]}"
  SEED="${run_values[1]}"
  LOG_DIR="${ARTIFACT_ROOT}/logs/${label}_$(date -u +%Y%m%dT%H%M%S%N)"
  mkdir -p "${LOG_DIR}"
  printf '%s\n' "${output}" >"${LOG_DIR}/run_context.log"
}

_run_cpu_logged() {
  local name="$1"
  shift
  local log_path="${LOG_DIR}/${name}.log"
  printf 'execute CPU: %s\n' "${name}"
  if ! _run_cpu "$@" 2>&1 | tee "${log_path}"; then
    printf 'CPU execution failed: %s\n' "${name}" >&2
    return 1
  fi
}

declare -a STAGE_DATASETS=()

_load_datasets() {
  local feature="$1"
  local output
  if ! output="$(
    CUDA_VISIBLE_DEVICES="" _engine \
      --phase list \
      --feature "${feature}" \
      2>&1
  )"; then
    printf '%s\n' "${output}" >"${LOG_DIR}/list_${feature}.log"
    printf '%s\n' "${output}" >&2
    return 1
  fi
  printf '%s\n' "${output}" >"${LOG_DIR}/list_${feature}.log"
  mapfile -t STAGE_DATASETS <<< "${output}"
  if (( ${#STAGE_DATASETS[@]} == 0 )); then
    printf 'no datasets configured for %s\n' "${feature}" >&2
    return 2
  fi
}

_gpu_is_free() {
  local gpu_id="$1"
  local used_mib
  if [[ "${WORKSHOP_ODI_SKIP_GPU_WAIT:-0}" == "1" ]]; then
    return 0
  fi
  used_mib="$(
    nvidia-smi --id="${gpu_id}" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk 'NR == 1 {print $1}'
  )" || return 1
  [[ "${used_mib}" =~ ^[0-9]+$ ]] || return 1
  (( used_mib <= GPU_FREE_MIB ))
}

_wait_for_gpu() {
  local gpu_id="$1"
  until _gpu_is_free "${gpu_id}"; do
    printf 'waiting for GPU %s\n' "${gpu_id}"
    sleep 5
  done
}

_run_gpu() {
  local gpu_id="$1"
  local feature="$2"
  local dataset="$3"
  local cache_path="${CACHE_ROOT}/physical_gpu_${gpu_id}"
  local -a command=(
    "${PYTHON_BIN}" -m workshop_jax.engine
    --phase execute
    --feature "${feature}"
    --artifact-root "${ARTIFACT_ROOT}"
    --seed "${SEED}"
    --dataset "${dataset}"
  )
  mkdir -p "${cache_path}/jax"
  if [[ -n "${CPU_AFFINITY}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
      XDG_CACHE_HOME="${cache_path}" \
      JAX_COMPILATION_CACHE_DIR="${cache_path}/jax" \
      OMP_NUM_THREADS="${IO_WORKERS}" \
      OPENBLAS_NUM_THREADS="${IO_WORKERS}" \
      MKL_NUM_THREADS="${IO_WORKERS}" \
      NUMEXPR_NUM_THREADS="${IO_WORKERS}" \
      taskset --cpu-list "${CPU_AFFINITY}" \
      "${command[@]}"
  else
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
      XDG_CACHE_HOME="${cache_path}" \
      JAX_COMPILATION_CACHE_DIR="${cache_path}/jax" \
      "${command[@]}"
  fi
}

_run_queue() {
  local gpu_id="$1"
  local feature="$2"
  shift 2
  local -a datasets=("$@")
  local status=0
  local log_path dataset
  for dataset in "${datasets[@]}"; do
    log_path="${LOG_DIR}/execute_${feature}_${dataset}.log"
    _wait_for_gpu "${gpu_id}"
    printf 'execute GPU %s: %s %s\n' "${gpu_id}" "${feature}" "${dataset}"
    if ! _run_gpu \
      "${gpu_id}" \
      "${feature}" \
      "${dataset}" \
      >"${log_path}" 2>&1; then
      status=1
      printf 'execute failed: %s %s\n' "${feature}" "${dataset}" >&2
      sed -n '1,240p' "${log_path}" >&2
    fi
  done
  return "${status}"
}

_run_pool() {
  local feature="$1"
  shift
  local status=0
  local slot index pid dataset
  local -a datasets=("$@")
  local -a pids=()
  local -a assigned=()
  local -a gpu_zero_order=(cifar10-dvs dvs128-gesture cifar-10_16)
  local -a gpu_one_order=(s-mnist shd cifar-100_16)
  if [[ "${feature}" == "train_regularization_on" ]]; then
    gpu_zero_order=(cifar10-dvs shd)
    gpu_one_order=(cifar-100_16 dvs128-gesture cifar-10_16 s-mnist)
  fi
  local fixed_two_gpu_schedule=0
  if (( ${#GPU_IDS[@]} == 2 )) && {
    [[ "${GPU_IDS[0]}" == "0" && "${GPU_IDS[1]}" == "1" ]] \
      || [[ "${GPU_IDS[0]}" == "1" && "${GPU_IDS[1]}" == "0" ]]
  }; then
    fixed_two_gpu_schedule=1
  fi
  if (( fixed_two_gpu_schedule == 1 )); then
    declare -A requested=()
    for dataset in "${datasets[@]}"; do
      requested["${dataset}"]=1
    done
    if (( ${#datasets[@]} != 6 || ${#requested[@]} != 6 )); then
      printf 'fixed GPU 0,1 schedule requires the six paper datasets\n' >&2
      return 2
    fi
    for dataset in "${gpu_zero_order[@]}" "${gpu_one_order[@]}"; do
      if [[ -z "${requested[${dataset}]:-}" ]]; then
        printf 'fixed GPU 0,1 schedule is missing dataset: %s\n' "${dataset}" >&2
        return 2
      fi
    done
    _run_queue "0" "${feature}" "${gpu_zero_order[@]}" &
    pids+=("$!")
    _run_queue "1" "${feature}" "${gpu_one_order[@]}" &
    pids+=("$!")
  else
    for slot in "${!GPU_IDS[@]}"; do
      assigned=()
      for ((index=slot; index<${#datasets[@]}; index+=${#GPU_IDS[@]})); do
        assigned+=("${datasets[index]}")
      done
      _run_queue \
        "${GPU_IDS[slot]}" \
        "${feature}" \
        "${assigned[@]}" &
      pids+=("$!")
    done
  fi
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  return "${status}"
}

_print_result() {
  printf 'paper artifacts: %s\n' "${ARTIFACT_ROOT}"
  printf 'execution logs: %s\n' "${LOG_DIR}"
}

paper_data_main() {
  _initialize_context \
    data \
    --phase initialize \
    --run-config "${RUN_CONFIG}"
  _run_cpu_logged \
    execute_data_prep \
    --phase execute \
    --feature data_prep \
    --artifact-root "${ARTIFACT_ROOT}" \
    --seed "${SEED}"
  _load_datasets dataset_signal_analysis
  _run_pool \
    dataset_signal_analysis \
    "${STAGE_DATASETS[@]}"
  _run_cpu_logged \
    finalize_dataset_signal_analysis \
    --phase finalize \
    --feature dataset_signal_analysis \
    --artifact-root "${ARTIFACT_ROOT}" \
    --seed "${SEED}"
  _print_result
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  paper_data_main "$@"
fi
