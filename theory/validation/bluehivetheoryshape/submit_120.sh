#!/bin/bash
set -euo pipefail

module load python3/3.11.10
source /home/kshao4/env/bin/activate

python3 generate_tasks.py
python3 generate_slurm_files.py
mkdir -p logs task_results

for job in slurm/shape_*.slurm; do
    sbatch "$job"
done
