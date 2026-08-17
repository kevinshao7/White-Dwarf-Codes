from __future__ import annotations

import argparse
import concurrent.futures
import os
import time
from pathlib import Path

from hpc_shape_common import (
    DEFAULT_CURVE_POINTS,
    DEFAULT_MAX_VELOCITY_CM_S,
    DEFAULT_MIN_VELOCITY_CM_S,
    DEFAULT_VRES,
    RESULTS_DIR,
    TASKS_CSV,
    add_shape_columns,
    read_rows_csv,
    run_condition_velocity_task,
    task_output_path,
    write_rows_csv,
)


def run_one(task: dict[str, str], gpu_count: int) -> list[dict[str, object]]:
    task_id = int(task["task_id"])
    if gpu_count > 0:
        task["gpu_id"] = str((task_id - 1) % gpu_count)
    rows = run_condition_velocity_task(task)
    write_rows_csv(task_output_path(task_id), rows)
    return rows


def load_or_generate_tasks(tasks_csv: Path) -> list[dict[str, str]]:
    if not tasks_csv.exists():
        from generate_tasks import main as generate_tasks_main

        generate_tasks_main()
    return read_rows_csv(tasks_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Dungeon shape sweep directly without scheduler jobs.")
    parser.add_argument("--tasks-csv", type=Path, default=TASKS_CSV)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("DUNGEON_CPU_CORES", "20")))
    parser.add_argument("--gpus", type=int, default=int(os.environ.get("DUNGEON_GPUS", "2")))
    parser.add_argument("--output-csv", type=Path, default=Path(__file__).resolve().parent / "dungeon_impact_parameter_shape_curves.csv")
    parser.add_argument("--curve-points", type=int, default=DEFAULT_CURVE_POINTS)
    parser.add_argument("--min-velocity-cm-s", type=float, default=DEFAULT_MIN_VELOCITY_CM_S)
    parser.add_argument("--max-velocity-cm-s", type=float, default=DEFAULT_MAX_VELOCITY_CM_S)
    parser.add_argument("--vres", type=int, default=DEFAULT_VRES)
    args = parser.parse_args()

    if not args.tasks_csv.exists():
        from generate_tasks import write_task_table

        write_task_table(
            output_csv=args.tasks_csv,
            curve_points=args.curve_points,
            min_velocity_cm_s=args.min_velocity_cm_s,
            max_velocity_cm_s=args.max_velocity_cm_s,
            vres=args.vres,
        )

    tasks = read_rows_csv(args.tasks_csv)
    if not tasks:
        raise SystemExit(f"No tasks found in {args.tasks_csv}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    all_rows: list[dict[str, object]] = []
    worker_count = max(1, min(args.workers, len(tasks)))
    gpu_count = max(0, args.gpus)

    print(
        f"[run start] tasks={len(tasks)} workers={worker_count} gpus={gpu_count} "
        f"tasks_csv={args.tasks_csv}",
        flush=True,
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(run_one, dict(task), gpu_count): int(task["task_id"]) for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            task_id = futures[future]
            rows = future.result()
            all_rows.extend(rows)
            print(f"[run progress] finished task_id={task_id} rows={len(rows)}", flush=True)

    all_rows.sort(
        key=lambda row: (
            int(row["condition"]),
            float(row["bmax_over_hydrogen_interparticle_spacing"]),
            float(row["velocity_cm_s"]),
        )
    )
    add_shape_columns(all_rows)
    write_rows_csv(args.output_csv, all_rows)
    elapsed = time.perf_counter() - started
    print(f"[run done] wrote {args.output_csv} rows={len(all_rows)} elapsed_min={elapsed / 60.0:.2f}")


if __name__ == "__main__":
    main()
