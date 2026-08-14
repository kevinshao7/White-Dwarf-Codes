#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

setup_bluehive_env() {
    module purge
    module load slurm || {
        echo "Failed to load the SLURM module. Available SLURM modules:"
        module avail slurm
        exit 1
    }
    module load python3/3.11.10
    source /home/kshao4/env/bin/activate
    python3 -c "import numpy, scipy"
}

setup_bluehive_env

if [[ $# -eq 0 && -z "${TASK_ID:-}" ]]; then
    command -v sbatch >/dev/null 2>&1 || {
        echo "sbatch not found after loading the SLURM module; run this from a BlueHive login node."
        exit 1
    }

    python3 generate_tasks.py
    python3 generate_slurm_files.py
    mkdir -p logs task_results

    for job in slurm/shape_*.slurm; do
        sbatch "$job"
    done
    exit 0
fi

mkdir -p logs task_results
python3 run_task.py "$@"
