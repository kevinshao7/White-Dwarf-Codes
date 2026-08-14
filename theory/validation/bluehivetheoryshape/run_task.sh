#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

module purge
module load python3/3.11.10
source /home/kshao4/env/bin/activate
python3 -c "import numpy, scipy"

mkdir -p logs task_results
python3 run_task.py "$@"
