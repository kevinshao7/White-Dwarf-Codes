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


def write_task_table(
    output_csv: Path = TASKS_CSV,
    conditions: list[int] | tuple[int, ...] = CONDITIONS,
    curve_points: int = DEFAULT_CURVE_POINTS,
    min_velocity_cm_s: float = DEFAULT_MIN_VELOCITY_CM_S,
    max_velocity_cm_s: float = DEFAULT_MAX_VELOCITY_CM_S,
    vres: int = DEFAULT_VRES,
) -> None:
    velocities = log_velocity_grid(min_velocity_cm_s, max_velocity_cm_s, curve_points)
    rows = []
    task_id = 1
    for condition in conditions:
        for velocity_cm_s in velocities:
            rows.append(
                {
                    "task_id": task_id,
                    "condition": int(condition),
                    "velocity_cm_s": float(velocity_cm_s),
                    "vres": int(vres),
                }
            )
            task_id += 1

    write_rows_csv(output_csv, rows)
    print(f"Wrote {output_csv}")
    print(f"Generated {len(rows)} condition-velocity tasks.")
    print(f"Each task computes bmax/aH={list(BMAX_OVER_AH)} from one shared grid.")
    print(
        f"Shared grid: {IMPACT_GRID} launch-impact spacing, rhores={SHARED_RHORES}, "
        f"max bmax/aH={SHARED_BMAX_OVER_AH:g}, min/max={MIN_IMPACT_OVER_MAX:g}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Dungeon task table for impact-parameter shape sweep.")
    parser.add_argument("--conditions", nargs="+", type=int, default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--curve-points", type=int, default=DEFAULT_CURVE_POINTS)
    parser.add_argument("--min-velocity-cm-s", type=float, default=DEFAULT_MIN_VELOCITY_CM_S)
    parser.add_argument("--max-velocity-cm-s", type=float, default=DEFAULT_MAX_VELOCITY_CM_S)
    parser.add_argument("--vres", type=int, default=DEFAULT_VRES)
    parser.add_argument("--output-csv", type=Path, default=TASKS_CSV)
    args = parser.parse_args()

    write_task_table(
        output_csv=args.output_csv,
        conditions=args.conditions,
        curve_points=args.curve_points,
        min_velocity_cm_s=args.min_velocity_cm_s,
        max_velocity_cm_s=args.max_velocity_cm_s,
        vres=args.vres,
    )


if __name__ == "__main__":
    main()
