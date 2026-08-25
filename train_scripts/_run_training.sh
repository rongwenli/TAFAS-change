#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 MODEL DATASET PRED_LEN [YACS_OVERRIDE ...]" >&2
    exit 2
fi

MODEL="$1"
DATASET="$2"
PRED_LEN="$3"
shift 3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${TAFAS_PYTHON:-python}"
CHECKPOINT_ROOT="${TAFAS_CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
RESULT_ROOT="${TAFAS_RESULT_ROOT:-${PROJECT_ROOT}/results}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${MODEL}/${DATASET}_${PRED_LEN}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/checkpoint_best.pth"
RESULT_DIR="${RESULT_ROOT}/${MODEL}/${DATASET}_${PRED_LEN}/train"
DATA_PATH="${PROJECT_ROOT}/data/${DATASET}/${DATASET}.csv"

cd "${PROJECT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Error: Python executable not found: ${PYTHON_BIN}" >&2
    echo "Activate the tafas conda environment, or set TAFAS_PYTHON explicitly." >&2
    exit 1
fi

if [[ ! -r "${DATA_PATH}" ]]; then
    echo "Error: dataset file is not readable: ${DATA_PATH}" >&2
    echo "Expected layout: data/${DATASET}/${DATASET}.csv" >&2
    exit 1
fi

mkdir -p "${CHECKPOINT_DIR}" "${RESULT_DIR}"

echo "Training ${MODEL} on ${DATASET} (prediction length: ${PRED_LEN})"
echo "Checkpoint output: ${CHECKPOINT_PATH}"

"${PYTHON_BIN}" main.py \
    DATA.NAME "${DATASET}" \
    DATA.PRED_LEN "${PRED_LEN}" \
    MODEL.NAME "${MODEL}" \
    MODEL.pred_len "${PRED_LEN}" \
    TRAIN.ENABLE True \
    TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}/" \
    TEST.ENABLE False \
    TTA.ENABLE False \
    RESULT_DIR "${RESULT_DIR}/" \
    "$@"

if [[ ! -s "${CHECKPOINT_PATH}" ]]; then
    echo "Error: training completed without creating ${CHECKPOINT_PATH}" >&2
    exit 1
fi

echo "Training complete: ${CHECKPOINT_PATH}"
