from __future__ import annotations

import argparse
from pathlib import Path

from hpc_shape_common import SLURM_DIR, TASKS_CSV, read_rows_csv


SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --partition=standard
#SBATCH --time={time_limit}
#SBATCH --mail-type=END
#SBATCH --mail-user={mail_user}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o logs/shape_{task_id:03d}.out
#SBATCH -e logs/shape_{task_id:03d}.err

set -euo pipefail

cd "${{SLURM_SUBMIT_DIR}}"
mkdir -p logs task_results

export TASK_ID={task_id}
bash ./run_task.sh --task-id "${{TASK_ID}}"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one 1-CPU SLURM file per condition-velocity task.")
    parser.add_argument("--tasks-csv", type=Path, default=TASKS_CSV)
    parser.add_argument("--output-dir", type=Path, default=SLURM_DIR)
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--mail-user", default="ks2120@cam.ac.uk")
    args = parser.parse_args()

    tasks = read_rows_csv(args.tasks_csv)
    if len(tasks) != 120:
        raise SystemExit(f"Expected 120 tasks, found {len(tasks)} in {args.tasks_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        task_id = int(task["task_id"])
        path = args.output_dir / f"shape_{task_id:03d}.slurm"
        path.write_text(
            SLURM_TEMPLATE.format(task_id=task_id, time_limit=args.time, mail_user=args.mail_user),
            encoding="utf-8",
            newline="\n",
        )
    print(f"Wrote {len(tasks)} SLURM files to {args.output_dir}")


if __name__ == "__main__":
    main()
