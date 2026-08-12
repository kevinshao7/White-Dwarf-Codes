#!/bin/bash
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ks2120@cam.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o logs/shape_117.out
#SBATCH -e logs/shape_117.err

set -euo pipefail

module load python3/3.12.0
source /home/kshao4/env/bin/activate

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs task_results

export TASK_ID=117
python run_task.py --task-id "${TASK_ID}"
