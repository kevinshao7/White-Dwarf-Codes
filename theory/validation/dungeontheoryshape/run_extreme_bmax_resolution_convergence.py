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
    DEFAULT_VRES,
    OUTDIR,
    SHARED_RHORES,
    condition_label,
    run_condition_velocity_task,
    write_rows_csv,
)


DEFAULT_FACTORS = (0.1, 0.3, 1.0, 3.0, 10.0)
DEFAULT_CONDITIONS = (0, 1, 2, 3)
EXTREME_BMAX_OVER_AH = (0.1, 10.0)


def scaled_resolution(factor: float) -> dict[str, int]:
    if factor <= 0.0:
        raise ValueError("resolution scale factors must be positive")
    return {
        "vres": max(3, int(round(DEFAULT_VRES * factor))),
        "rhores": max(10, int(round(SHARED_RHORES * factor))),
        "ures": max(8, int(round(BASE_UDPHIRES * factor))),
        "dphires": max(8, int(round(BASE_UDPHIRES * factor))),
    }


def finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def run_condition_factor(
    task_id: int,
    condition: int,
    factor: float,
    velocity_cm_s: float,
    gpu_count: int,
) -> list[dict[str, object]]:
    resolution = scaled_resolution(factor)
    gpu_id = (task_id - 1) % gpu_count if gpu_count > 0 else -1
    task = {
        "task_id": task_id,
        "condition": condition,
        "condition_label": condition_label(condition),
        "velocity_cm_s": velocity_cm_s,
        "gpu_id": gpu_id,
        **resolution,
    }
    print(
        "[extreme convergence start] "
        f"task_id={task_id} condition={condition} factor={factor:g} "
        f"velocity_cm_s={velocity_cm_s:.6e} vres={resolution['vres']} "
        f"rhores={resolution['rhores']} ures={resolution['ures']} "
        f"dphires={resolution['dphires']} gpu_id={gpu_id if gpu_id >= 0 else 'cpu'}",
        flush=True,
    )
    rows = run_condition_velocity_task(task)
    filtered = []
    for row in rows:
        bmax = finite_float(row["bmax_over_hydrogen_interparticle_spacing"])
        if any(math.isclose(bmax, target, rel_tol=0.0, abs_tol=1.0e-12) for target in EXTREME_BMAX_OVER_AH):
            row["resolution_factor"] = factor
            row["reference_condition_label"] = condition_label(condition)
            filtered.append(row)
    return filtered


def plot_convergence(rows: list[dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.0), sharex=True)
    axes_flat = axes.ravel()
    colors = {0.1: "#1f77b4", 10.0: "#d62728"}

    for axis, condition in zip(axes_flat, DEFAULT_CONDITIONS):
        condition_rows = [row for row in rows if int(row["condition"]) == condition]
        highest_factor = max(finite_float(row["resolution_factor"]) for row in condition_rows)
        reference_by_bmax = {
            finite_float(row["bmax_over_hydrogen_interparticle_spacing"]): finite_float(row["absolute_drag_N"])
            for row in condition_rows
            if finite_float(row["resolution_factor"]) == highest_factor
        }
        for bmax in EXTREME_BMAX_OVER_AH:
            curve = sorted(
                (
                    row
                    for row in condition_rows
                    if math.isclose(
                        finite_float(row["bmax_over_hydrogen_interparticle_spacing"]),
                        bmax,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                ),
                key=lambda row: finite_float(row["resolution_factor"]),
            )
            factors = np.array([finite_float(row["resolution_factor"]) for row in curve], dtype=float)
            drag = np.array([finite_float(row["absolute_drag_N"]) for row in curve], dtype=float)
            reference = reference_by_bmax.get(bmax, math.nan)
            relative = np.abs(drag / reference - 1.0) if math.isfinite(reference) and reference != 0.0 else np.nan * drag
            axis.plot(
                factors,
                relative,
                marker="o",
                linewidth=1.8,
                markersize=4.0,
                color=colors[bmax],
                label=rf"$b_{{max}}/a_H={bmax:g}$",
            )

        axis.set_title(f"Condition {condition}: {condition_label(condition)}")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("resolution scale factor")
        axis.set_ylabel("relative difference from 10x")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)

    first = rows[0]
    fig.suptitle(
        "Dungeon extreme-bmax resolution convergence "
        f"at v={finite_float(first['velocity_cm_s']):.6g} cm/s"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check convergence for bmax/aH=0.1 and 10 across all Dungeon conditions."
    )
    parser.add_argument("--velocity-cm-s", type=float, default=1.0e8)
    parser.add_argument("--conditions", nargs="+", type=int, default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--factors", default=",".join(f"{factor:g}" for factor in DEFAULT_FACTORS))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("DUNGEON_CPU_CORES", "20")))
    parser.add_argument("--gpus", type=int, default=int(os.environ.get("DUNGEON_GPUS", "2")))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTDIR / "extreme_bmax_resolution_convergence.csv",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=OUTDIR / "extreme_bmax_resolution_convergence.png",
    )
    args = parser.parse_args()

    factors = [float(value) for value in args.factors.split(",") if value.strip()]
    jobs = []
    task_id = 1
    for condition in args.conditions:
        for factor in factors:
            jobs.append((task_id, condition, factor))
            task_id += 1
    worker_count = max(1, min(args.workers, len(jobs)))
    gpu_count = max(0, args.gpus)
    print(
        "[extreme convergence run] "
        f"conditions={args.conditions} factors={factors} workers={worker_count} "
        f"gpus={gpu_count} velocity_cm_s={args.velocity_cm_s:.6e}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(run_condition_factor, task_id, condition, factor, args.velocity_cm_s, gpu_count): (
                condition,
                factor,
            )
            for task_id, condition, factor in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            condition, factor = futures[future]
            factor_rows = future.result()
            rows.extend(factor_rows)
            print(
                "[extreme convergence progress] "
                f"finished condition={condition} factor={factor:g} rows={len(factor_rows)}",
                flush=True,
            )

    rows.sort(
        key=lambda row: (
            int(row["condition"]),
            float(row["bmax_over_hydrogen_interparticle_spacing"]),
            float(row["resolution_factor"]),
        )
    )
    write_rows_csv(args.output_csv, rows)
    plot_convergence(rows, args.output_png)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
