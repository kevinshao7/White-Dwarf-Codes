from __future__ import annotations

import argparse
import os
from pathlib import Path

from hpc_shape_common import TASKS_CSV, read_rows_csv, run_condition_velocity_task, task_output_path, write_rows_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one BlueHive impact-parameter shape task.")
    parser.add_argument("--tasks-csv", type=Path, default=TASKS_CSV)
    parser.add_argument(
        "--task-id",
        type=int,
        default=int(os.environ.get("TASK_ID", "0")),
        help="1-based task id. Defaults to TASK_ID.",
    )
    args = parser.parse_args()

    if args.task_id < 1:
        raise SystemExit("Provide --task-id or set TASK_ID in the SLURM file.")

    tasks = read_rows_csv(args.tasks_csv)
    if args.task_id > len(tasks):
        raise SystemExit(f"task id {args.task_id} exceeds task count {len(tasks)}")

    task = tasks[args.task_id - 1]
    if int(task["task_id"]) != args.task_id:
        raise SystemExit(f"Task table row mismatch: expected task_id={args.task_id}, got {task['task_id']}")

    rows = run_condition_velocity_task(task)
    output = task_output_path(args.task_id)
    write_rows_csv(output, rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
