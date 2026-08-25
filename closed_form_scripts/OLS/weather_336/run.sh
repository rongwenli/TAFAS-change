#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

exec bash "${PROJECT_ROOT}/closed_form_scripts/_run_closed_form.sh" \
    "OLS" "weather" "336" "$@"
