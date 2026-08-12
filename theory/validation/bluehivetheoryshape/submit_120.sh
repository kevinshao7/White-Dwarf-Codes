#!/bin/bash
set -euo pipefail

module load python3/3.12.0
source /home/kshao4/env/bin/activate

python generate_tasks.py
python generate_slurm_files.py
mkdir -p logs task_results

for job in slurm/shape_*.slurm; do
    sbatch "$job"
done
