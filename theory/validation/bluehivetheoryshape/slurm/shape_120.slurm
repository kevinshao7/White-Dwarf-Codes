#!/bin/bash
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ks2120@cam.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o logs/shape_120.out
#SBATCH -e logs/shape_120.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs task_results

export TASK_ID=120
bash ./run_task.sh --task-id "${TASK_ID}"
