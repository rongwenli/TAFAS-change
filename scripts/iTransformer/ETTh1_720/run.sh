#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${TAFAS_PYTHON:-python}"
CHECKPOINT_ROOT="${TAFAS_CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
RESULT_ROOT="${TAFAS_RESULT_ROOT:-${PROJECT_ROOT}/results}"

DATASET="ETTh1"
PRED_LEN=720
MODEL="iTransformer"
BASE_LR=0.001
WEIGHT_DECAY=0.0
GATING_INIT=0.01

CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${MODEL}/${DATASET}_${PRED_LEN}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/checkpoint_best.pth"
TRAIN_SCRIPT="${PROJECT_ROOT}/train_scripts/${MODEL}/${DATASET}_${PRED_LEN}/run.sh"
RESULT_DIR="${RESULT_ROOT}/${MODEL}/${DATASET}_${PRED_LEN}/tta"
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

if [[ ! -s "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT_PATH}"
    echo "Running the matching training script first: ${TRAIN_SCRIPT}"
    bash "${TRAIN_SCRIPT}"
fi

if [[ ! -s "${CHECKPOINT_PATH}" ]]; then
    echo "Error: checkpoint is still missing after training: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

echo "Using checkpoint: ${CHECKPOINT_PATH}"

"${PYTHON_BIN}" main.py \
    DATA.NAME "${DATASET}" \
    DATA.PRED_LEN "${PRED_LEN}" \
    MODEL.NAME "${MODEL}" \
    MODEL.pred_len "${PRED_LEN}" \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}/" \
    TTA.ENABLE True \
    TTA.SOLVER.BASE_LR "${BASE_LR}" \
    TTA.SOLVER.WEIGHT_DECAY "${WEIGHT_DECAY}" \
    TTA.TAFAS.GATING_INIT "${GATING_INIT}" \
    RESULT_DIR "${RESULT_DIR}/" \
    "$@"
