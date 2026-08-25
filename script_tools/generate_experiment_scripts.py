#!/usr/bin/env python3
"""Generate training, TAFAS, and closed-form experiment script triplets."""

from __future__ import annotations

import argparse
import re
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TTA_ROOT = ROOT / "scripts"
TRAIN_ROOT = ROOT / "train_scripts"
CLOSED_FORM_ROOT = ROOT / "closed_form_scripts"
REQUIRED_VALUES = (
    "DATASET",
    "PRED_LEN",
    "MODEL",
    "BASE_LR",
    "WEIGHT_DECAY",
    "GATING_INIT",
)


def read_value(text: str, name: str) -> str:
    match = re.search(rf"^{name}=(?:\"([^\"]*)\"|([^\s#]+))\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Missing {name}")
    return match.group(1) if match.group(1) is not None else match.group(2)


def render_training_wrapper(model: str, dataset: str, pred_len: str) -> str:
    return f'''#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_ROOT="$(cd -- "${{SCRIPT_DIR}}/../../.." && pwd)"

exec bash "${{PROJECT_ROOT}}/train_scripts/_run_training.sh" \\
    "{model}" "{dataset}" "{pred_len}" "$@"
'''


def render_closed_form_wrapper(model: str, dataset: str, pred_len: str) -> str:
    return f'''#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_ROOT="$(cd -- "${{SCRIPT_DIR}}/../../.." && pwd)"

exec bash "${{PROJECT_ROOT}}/closed_form_scripts/_run_closed_form.sh" \\
    "{model}" "{dataset}" "{pred_len}" "$@"
'''


def render_tta(values: dict[str, str]) -> str:
    return f'''#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_ROOT="$(cd -- "${{SCRIPT_DIR}}/../../.." && pwd)"
PYTHON_BIN="${{TAFAS_PYTHON:-python}}"
CHECKPOINT_ROOT="${{TAFAS_CHECKPOINT_ROOT:-${{PROJECT_ROOT}}/checkpoints}}"
RESULT_ROOT="${{TAFAS_RESULT_ROOT:-${{PROJECT_ROOT}}/results}}"

DATASET="{values['DATASET']}"
PRED_LEN={values['PRED_LEN']}
MODEL="{values['MODEL']}"
BASE_LR={values['BASE_LR']}
WEIGHT_DECAY={values['WEIGHT_DECAY']}
GATING_INIT={values['GATING_INIT']}

CHECKPOINT_DIR="${{CHECKPOINT_ROOT}}/${{MODEL}}/${{DATASET}}_${{PRED_LEN}}"
CHECKPOINT_PATH="${{CHECKPOINT_DIR}}/checkpoint_best.pth"
TRAIN_SCRIPT="${{PROJECT_ROOT}}/train_scripts/${{MODEL}}/${{DATASET}}_${{PRED_LEN}}/run.sh"
RESULT_DIR="${{RESULT_ROOT}}/${{MODEL}}/${{DATASET}}_${{PRED_LEN}}/tta"
DATA_PATH="${{PROJECT_ROOT}}/data/${{DATASET}}/${{DATASET}}.csv"

cd "${{PROJECT_ROOT}}"

if ! command -v "${{PYTHON_BIN}}" >/dev/null 2>&1; then
    echo "Error: Python executable not found: ${{PYTHON_BIN}}" >&2
    echo "Activate the tafas conda environment, or set TAFAS_PYTHON explicitly." >&2
    exit 1
fi

if [[ ! -r "${{DATA_PATH}}" ]]; then
    echo "Error: dataset file is not readable: ${{DATA_PATH}}" >&2
    echo "Expected layout: data/${{DATASET}}/${{DATASET}}.csv" >&2
    exit 1
fi

mkdir -p "${{CHECKPOINT_DIR}}" "${{RESULT_DIR}}"

if [[ ! -s "${{CHECKPOINT_PATH}}" ]]; then
    echo "Checkpoint not found: ${{CHECKPOINT_PATH}}"
    echo "Running the matching training script first: ${{TRAIN_SCRIPT}}"
    bash "${{TRAIN_SCRIPT}}"
fi

if [[ ! -s "${{CHECKPOINT_PATH}}" ]]; then
    echo "Error: checkpoint is still missing after training: ${{CHECKPOINT_PATH}}" >&2
    exit 1
fi

echo "Using checkpoint: ${{CHECKPOINT_PATH}}"

"${{PYTHON_BIN}}" main.py \\
    DATA.NAME "${{DATASET}}" \\
    DATA.PRED_LEN "${{PRED_LEN}}" \\
    MODEL.NAME "${{MODEL}}" \\
    MODEL.pred_len "${{PRED_LEN}}" \\
    TRAIN.ENABLE False \\
    TRAIN.CHECKPOINT_DIR "${{CHECKPOINT_DIR}}/" \\
    TTA.ENABLE True \\
    TTA.SOLVER.BASE_LR "${{BASE_LR}}" \\
    TTA.SOLVER.WEIGHT_DECAY "${{WEIGHT_DECAY}}" \\
    TTA.TAFAS.GATING_INIT "${{GATING_INIT}}" \\
    RESULT_DIR "${{RESULT_DIR}}/" \\
    "$@"
'''


def compare_or_write(path: Path, expected: str, check: bool) -> bool:
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == expected:
        return True
    if check:
        print(f"out of date: {path.relative_to(ROOT)}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"updated: {path.relative_to(ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="only report stale scripts")
    args = parser.parse_args()

    tta_scripts = sorted(TTA_ROOT.glob("*/*/run.sh"))
    if not tta_scripts:
        raise SystemExit("No TTA scripts found")

    all_current = True
    for tta_path in tta_scripts:
        original = tta_path.read_text(encoding="utf-8")
        try:
            values = {name: read_value(original, name) for name in REQUIRED_VALUES}
        except ValueError as exc:
            raise SystemExit(f"{tta_path}: {exc}") from exc

        relative = tta_path.relative_to(TTA_ROOT)
        expected_directory = Path(values["MODEL"]) / f'{values["DATASET"]}_{values["PRED_LEN"]}' / "run.sh"
        if relative != expected_directory:
            raise SystemExit(
                f"Path/config mismatch: {relative} != {expected_directory}"
            )

        train_path = TRAIN_ROOT / relative
        closed_form_path = CLOSED_FORM_ROOT / relative
        train_text = render_training_wrapper(
            values["MODEL"], values["DATASET"], values["PRED_LEN"]
        )
        closed_form_text = render_closed_form_wrapper(
            values["MODEL"], values["DATASET"], values["PRED_LEN"]
        )
        tta_text = render_tta(values)
        all_current &= compare_or_write(train_path, train_text, args.check)
        all_current &= compare_or_write(
            closed_form_path, closed_form_text, args.check
        )
        all_current &= compare_or_write(tta_path, tta_text, args.check)

    if args.check:
        print(f"checked {len(tta_scripts)} experiment script triplets")
    else:
        print(f"generated {len(tta_scripts)} experiment script triplets")
    return 0 if all_current else 1


if __name__ == "__main__":
    raise SystemExit(main())
