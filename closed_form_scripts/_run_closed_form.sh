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
TRAIN_SCRIPT="${PROJECT_ROOT}/train_scripts/${MODEL}/${DATASET}_${PRED_LEN}/run.sh"
RESULT_DIR="${RESULT_ROOT}/${MODEL}/${DATASET}_${PRED_LEN}/closed_form_tta"
DATA_PATH="${PROJECT_ROOT}/data/${DATASET}/${DATASET}.csv"

RANK="${CLOSED_FORM_RANK:-8}"
PREDICTION_MEMORY_SIZE="${CLOSED_FORM_PREDICTION_MEMORY_SIZE:-64}"
SUPERVISION_BUFFER_SIZE="${CLOSED_FORM_SUPERVISION_BUFFER_SIZE:-32}"
PCA_NORMALIZATION="${CLOSED_FORM_PCA_NORMALIZATION:-per_trajectory}"
MIN_PCA_SAMPLES="${CLOSED_FORM_MIN_PCA_SAMPLES:-16}"
SUBSPACE_UPDATE_INTERVAL="${CLOSED_FORM_SUBSPACE_UPDATE_INTERVAL:-8}"
RIDGE_LAMBDA="${CLOSED_FORM_RIDGE_LAMBDA:-0.001}"
FORGETTING_FACTOR="${CLOSED_FORM_FORGETTING_FACTOR:-0.95}"
POGT_LEN="${CLOSED_FORM_POGT_LEN:-24}"

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
echo "Running Prediction-Derived Closed-Form TTA"

"${PYTHON_BIN}" main.py \
    DATA.NAME "${DATASET}" \
    DATA.PRED_LEN "${PRED_LEN}" \
    MODEL.NAME "${MODEL}" \
    MODEL.pred_len "${PRED_LEN}" \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}/" \
    TEST.ENABLE False \
    TTA.ENABLE True \
    TTA.METHOD CLOSED_FORM \
    TTA.CLOSED_FORM.RANK "${RANK}" \
    TTA.CLOSED_FORM.PREDICTION_MEMORY_SIZE "${PREDICTION_MEMORY_SIZE}" \
    TTA.CLOSED_FORM.SUPERVISION_BUFFER_SIZE "${SUPERVISION_BUFFER_SIZE}" \
    TTA.CLOSED_FORM.PCA_NORMALIZATION "${PCA_NORMALIZATION}" \
    TTA.CLOSED_FORM.MIN_PCA_SAMPLES "${MIN_PCA_SAMPLES}" \
    TTA.CLOSED_FORM.SUBSPACE_UPDATE_INTERVAL "${SUBSPACE_UPDATE_INTERVAL}" \
    TTA.CLOSED_FORM.RIDGE_LAMBDA "${RIDGE_LAMBDA}" \
    TTA.CLOSED_FORM.FORGETTING_FACTOR "${FORGETTING_FACTOR}" \
    TTA.CLOSED_FORM.POGT_LEN "${POGT_LEN}" \
    RESULT_DIR "${RESULT_DIR}/" \
    "$@"
