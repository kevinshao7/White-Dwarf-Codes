from __future__ import annotations

import argparse
from pathlib import Path

from hpc_shape_common import (
    CONDITIONS,
    BMAX_OVER_AH,
    DEFAULT_CURVE_POINTS,
    DEFAULT_MAX_VELOCITY_CM_S,
    DEFAULT_MIN_VELOCITY_CM_S,
    DEFAULT_VRES,
    IMPACT_GRID,
    MIN_IMPACT_OVER_MAX,
    SHARED_BMAX_OVER_AH,
    SHARED_RHORES,
    TASKS_CSV,
    log_velocity_grid,
    write_rows_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BlueHive task table for impact-parameter shape sweep.")
    parser.add_argument("--conditions", nargs="+", type=int, default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--curve-points", type=int, default=DEFAULT_CURVE_POINTS)
    parser.add_argument("--min-velocity-cm-s", type=float, default=DEFAULT_MIN_VELOCITY_CM_S)
    parser.add_argument("--max-velocity-cm-s", type=float, default=DEFAULT_MAX_VELOCITY_CM_S)
    parser.add_argument("--vres", type=int, default=DEFAULT_VRES)
    parser.add_argument("--output-csv", type=Path, default=TASKS_CSV)
    args = parser.parse_args()

    velocities = log_velocity_grid(args.min_velocity_cm_s, args.max_velocity_cm_s, args.curve_points)
    rows = []
    task_id = 1
    for condition in args.conditions:
        for velocity_cm_s in velocities:
            rows.append(
                {
                    "task_id": task_id,
                    "condition": int(condition),
                    "velocity_cm_s": float(velocity_cm_s),
                    "vres": int(args.vres),
                }
            )
            task_id += 1

    write_rows_csv(args.output_csv, rows)
    print(f"Wrote {args.output_csv}")
    print(f"Generated {len(rows)} condition-velocity tasks.")
    print(f"Each task computes bmax/aH={list(BMAX_OVER_AH)} from one shared grid.")
    print(
        f"Shared grid: {IMPACT_GRID} launch-impact spacing, rhores={SHARED_RHORES}, "
        f"max bmax/aH={SHARED_BMAX_OVER_AH:g}, min/max={MIN_IMPACT_OVER_MAX:g}."
    )


if __name__ == "__main__":
    main()
