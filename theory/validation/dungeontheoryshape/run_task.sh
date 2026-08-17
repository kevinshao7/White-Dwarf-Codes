#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

python3 -c "import numpy, scipy"

if [[ $# -eq 0 ]]; then
    python3 generate_tasks.py
    python3 run_all.py --workers "${DUNGEON_CPU_CORES:-20}" --gpus "${DUNGEON_GPUS:-2}"
    python3 aggregate_results.py
else
    mkdir -p task_results
    python3 run_task.py "$@"
fi
