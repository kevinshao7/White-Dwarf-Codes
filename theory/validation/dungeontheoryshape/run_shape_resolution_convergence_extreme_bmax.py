from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hpc_shape_common import (
    BASE_UDPHIRES,
    DEFAULT_CURVE_POINTS,
    DEFAULT_MAX_VELOCITY_CM_S,
    DEFAULT_MIN_VELOCITY_CM_S,
    DEFAULT_VRES,
    OUTDIR,
    SHARED_RHORES,
    condition_label,
    log_velocity_grid,
    require_usable_gpus,
    run_condition_velocity_task,
    write_rows_csv,
)


DEFAULT_FACTORS = (0.1, 0.3, 1.0, 3.0)
DEFAULT_CONDITIONS = (0,)
EXTREME_BMAX_OVER_AH = (0.1, 10.0)
DEFAULT_RESOLUTION_AXES = ("vres", "rhores", "angle")
DEFAULT_CONVERGENCE_CURVE_POINTS = 10


def scaled_resolution(factor: float, axis: str) -> dict[str, int]:
    if factor <= 0.0:
        raise ValueError("resolution scale factors must be positive")
    resolution = {
        "vres": DEFAULT_VRES,
        "rhores": SHARED_RHORES,
        "ures": BASE_UDPHIRES,
        "dphires": BASE_UDPHIRES,
    }
    if axis == "all":
        resolution["vres"] = max(3, int(round(DEFAULT_VRES * factor)))
        resolution["rhores"] = max(10, int(round(SHARED_RHORES * factor)))
        resolution["ures"] = max(8, int(round(BASE_UDPHIRES * factor)))
        resolution["dphires"] = max(8, int(round(BASE_UDPHIRES * factor)))
    elif axis == "vres":
        resolution["vres"] = max(3, int(round(DEFAULT_VRES * factor)))
    elif axis == "rhores":
        resolution["rhores"] = max(10, int(round(SHARED_RHORES * factor)))
    elif axis == "angle":
        resolution["ures"] = max(8, int(round(BASE_UDPHIRES * factor)))
        resolution["dphires"] = max(8, int(round(BASE_UDPHIRES * factor)))
    else:
        raise ValueError(f"unknown resolution axis {axis!r}")
    return resolution


def finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def is_extreme_bmax(value: object) -> bool:
    bmax = finite_float(value)
    return any(math.isclose(bmax, target, rel_tol=0.0, abs_tol=1.0e-12) for target in EXTREME_BMAX_OVER_AH)


def run_curve_point(
    task_id: int,
    condition: int,
    velocity_cm_s: float,
    factor: float,
    resolution_axis: str,
    gpu_count: int,
) -> list[dict[str, object]]:
    resolution = scaled_resolution(factor, resolution_axis)
    gpu_id = (task_id - 1) % gpu_count if gpu_count > 0 else -1
    task = {
        "task_id": task_id,
        "condition": condition,
        "condition_label": condition_label(condition),
        "velocity_cm_s": velocity_cm_s,
        "gpu_id": gpu_id,
        "require_gpu": gpu_count > 0,
        **resolution,
    }
    print(
        "[shape convergence start] "
        f"task_id={task_id} condition={condition} axis={resolution_axis} factor={factor:g} "
        f"velocity_cm_s={velocity_cm_s:.6e} vres={resolution['vres']} "
        f"rhores={resolution['rhores']} ures={resolution['ures']} "
        f"dphires={resolution['dphires']} gpu_id={gpu_id if gpu_id >= 0 else 'cpu'}",
        flush=True,
    )
    rows = run_condition_velocity_task(task)
    filtered = []
    for row in rows:
        if is_extreme_bmax(row["bmax_over_hydrogen_interparticle_spacing"]):
            row["resolution_factor"] = factor
            row["resolution_axis"] = resolution_axis
            row["reference_condition_label"] = condition_label(condition)
            filtered.append(row)
    return filtered


def add_curve_reference_columns(rows: list[dict[str, object]]) -> None:
    reference: dict[tuple[int, str, float, float], float] = {}
    highest_factor_by_axis = {
        str(axis): max(
            finite_float(row["resolution_factor"])
            for row in rows
            if str(row.get("resolution_axis", "all")) == str(axis)
        )
        for axis in {str(row.get("resolution_axis", "all")) for row in rows}
    }
    for row in rows:
        axis = str(row.get("resolution_axis", "all"))
        highest_factor = highest_factor_by_axis[axis]
        if finite_float(row["resolution_factor"]) != highest_factor:
            continue
        key = (
            int(row["condition"]),
            axis,
            finite_float(row["bmax_over_hydrogen_interparticle_spacing"]),
            finite_float(row["velocity_cm_s"]),
        )
        reference[key] = finite_float(row["absolute_drag_N"])

    for row in rows:
        key = (
            int(row["condition"]),
            str(row.get("resolution_axis", "all")),
            finite_float(row["bmax_over_hydrogen_interparticle_spacing"]),
            finite_float(row["velocity_cm_s"]),
        )
        ref = reference.get(key, math.nan)
        value = finite_float(row["absolute_drag_N"])
        row["reference_resolution_factor"] = highest_factor_by_axis[str(row.get("resolution_axis", "all"))]
        row["absolute_drag_N_at_reference_resolution"] = ref
        row["relative_error_vs_reference_resolution"] = abs(value / ref - 1.0) if math.isfinite(ref) and ref != 0.0 else math.nan


def plot_condition_axis(rows: list[dict[str, object]], condition: int, resolution_axis: str, output: Path) -> None:
    condition_rows = [row for row in rows if int(row["condition"]) == condition]
    condition_rows = [row for row in condition_rows if str(row.get("resolution_axis", "all")) == resolution_axis]
    if not condition_rows:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.0), sharex=True)
    factors = sorted({finite_float(row["resolution_factor"]) for row in condition_rows})
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(factors)))
    color_by_factor = dict(zip(factors, colors))

    for column, bmax in enumerate(EXTREME_BMAX_OVER_AH):
        bmax_rows = [
            row
            for row in condition_rows
            if math.isclose(
                finite_float(row["bmax_over_hydrogen_interparticle_spacing"]),
                bmax,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ]
        for factor in factors:
            curve = sorted(
                (row for row in bmax_rows if finite_float(row["resolution_factor"]) == factor),
                key=lambda row: finite_float(row["velocity_cm_s"]),
            )
            velocity = np.array([finite_float(row["velocity_cm_s"]) for row in curve], dtype=float)
            drag = np.array([finite_float(row["absolute_drag_N"]) for row in curve], dtype=float)
            relative = np.array([finite_float(row["relative_error_vs_reference_resolution"]) for row in curve], dtype=float)
            valid_drag = np.isfinite(velocity) & np.isfinite(drag) & (drag > 0.0)
            valid_relative = np.isfinite(velocity) & np.isfinite(relative) & (relative > 0.0)
            label = f"{factor:g}x"

            axes[0, column].plot(
                velocity[valid_drag],
                drag[valid_drag],
                marker="o",
                linewidth=1.6,
                markersize=3.2,
                color=color_by_factor[factor],
                label=label,
            )
            axes[1, column].plot(
                velocity[valid_relative],
                relative[valid_relative],
                marker="o",
                linewidth=1.6,
                markersize=3.2,
                color=color_by_factor[factor],
                label=label,
            )

        axes[0, column].set_title(rf"$b_{{max}}/a_H={bmax:g}$ absolute drag")
        axes[1, column].set_title(rf"$b_{{max}}/a_H={bmax:g}$ error vs 10x")
        axes[0, column].set_ylabel("|drag| [N]")
        axes[1, column].set_ylabel("relative error")
        axes[0, column].set_yscale("log")
        axes[1, column].set_yscale("log")

    for axis in axes.ravel():
        axis.set_xscale("log")
        axis.set_xlabel("velocity [cm/s]")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)

    fig.suptitle(f"Condition {condition}: {condition_label(condition)}, {resolution_axis} convergence")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check full velocity-shape convergence for bmax/aH=0.1 and 10 across Dungeon conditions."
    )
    parser.add_argument("--conditions", nargs="+", type=int, default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--factors", default=",".join(f"{factor:g}" for factor in DEFAULT_FACTORS))
    parser.add_argument("--curve-points", type=int, default=DEFAULT_CONVERGENCE_CURVE_POINTS)
    parser.add_argument("--min-velocity-cm-s", type=float, default=DEFAULT_MIN_VELOCITY_CM_S)
    parser.add_argument("--max-velocity-cm-s", type=float, default=DEFAULT_MAX_VELOCITY_CM_S)
    parser.add_argument(
        "--scale-mode",
        choices=("separate", "all"),
        default="separate",
        help="separate scales vres/rhores/angle one at a time; all scales every numerical dimension together.",
    )
    parser.add_argument("--axes", nargs="+", default=list(DEFAULT_RESOLUTION_AXES), choices=("vres", "rhores", "angle"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("DUNGEON_CPU_CORES", "20")))
    parser.add_argument("--gpus", type=int, default=int(os.environ.get("DUNGEON_GPUS", "2")))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTDIR / "shape_resolution_convergence_extreme_bmax.csv",
    )
    args = parser.parse_args()

    factors = [float(value) for value in args.factors.split(",") if value.strip()]
    velocities = log_velocity_grid(args.min_velocity_cm_s, args.max_velocity_cm_s, args.curve_points)
    resolution_axes = ["all"] if args.scale_mode == "all" else list(args.axes)
    jobs = []
    task_id = 1
    for condition in args.conditions:
        for resolution_axis in resolution_axes:
            for factor in factors:
                for velocity_cm_s in velocities:
                    jobs.append((task_id, condition, velocity_cm_s, factor, resolution_axis))
                    task_id += 1

    worker_count = max(1, min(args.workers, len(jobs)))
    gpu_count = max(0, args.gpus)
    require_usable_gpus(gpu_count)
    print(
        "[shape convergence run] "
        f"conditions={args.conditions} axes={resolution_axes} factors={factors} velocities={len(velocities)} "
        f"workers={worker_count} gpus={gpu_count}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(run_curve_point, task_id, condition, velocity_cm_s, factor, resolution_axis, gpu_count): (
                condition,
                velocity_cm_s,
                factor,
                resolution_axis,
            )
            for task_id, condition, velocity_cm_s, factor, resolution_axis in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            condition, velocity_cm_s, factor, resolution_axis = futures[future]
            curve_rows = future.result()
            rows.extend(curve_rows)
            print(
                "[shape convergence progress] "
                f"finished condition={condition} axis={resolution_axis} factor={factor:g} "
                f"velocity_cm_s={velocity_cm_s:.6e} rows={len(curve_rows)}",
                flush=True,
            )

    rows.sort(
        key=lambda row: (
            int(row["condition"]),
            str(row.get("resolution_axis", "all")),
            float(row["bmax_over_hydrogen_interparticle_spacing"]),
            float(row["resolution_factor"]),
            float(row["velocity_cm_s"]),
        )
    )
    add_curve_reference_columns(rows)
    write_rows_csv(args.output_csv, rows)
    print(f"Wrote {args.output_csv}")

    for condition in args.conditions:
        for resolution_axis in resolution_axes:
            output_png = OUTDIR / f"condition_{condition}_{resolution_axis}_shape_resolution_convergence_extreme_bmax.png"
            plot_condition_axis(rows, condition, resolution_axis, output_png)
            print(f"Wrote {output_png}")


if __name__ == "__main__":
    main()
