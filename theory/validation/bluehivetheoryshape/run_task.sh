#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ $# -eq 0 && -z "${TASK_ID:-}" ]]; then
    command -v sbatch >/dev/null 2>&1 || {
        echo "sbatch not found; run this from a BlueHive login node with SLURM available."
        exit 1
    }

    module purge
    module load python3/3.11.10
    source /home/kshao4/env/bin/activate
    python3 -c "import numpy, scipy"

    python3 generate_tasks.py
    python3 generate_slurm_files.py
    mkdir -p logs task_results

    for job in slurm/shape_*.slurm; do
        sbatch "$job"
    done
    exit 0
fi

module purge
module load python3/3.11.10
source /home/kshao4/env/bin/activate
python3 -c "import numpy, scipy"

mkdir -p logs task_results
python3 run_task.py "$@"
