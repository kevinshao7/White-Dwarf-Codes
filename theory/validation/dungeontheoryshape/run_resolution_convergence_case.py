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
    BMAX_OVER_AH,
    DEFAULT_VRES,
    OUTDIR,
    SHARED_RHORES,
    add_shape_columns,
    condition_label,
    run_condition_velocity_task,
    write_rows_csv,
)


DEFAULT_FACTORS = (0.1, 0.3, 1.0, 3.0, 10.0)


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


def plot_convergence(rows: list[dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(BMAX_OVER_AH)))
    reference_by_bmax = {}
    highest_factor = max(finite_float(row["resolution_factor"]) for row in rows)
    for row in rows:
        if finite_float(row["resolution_factor"]) == highest_factor:
            reference_by_bmax[finite_float(row["bmax_over_hydrogen_interparticle_spacing"])] = finite_float(
                row["absolute_drag_N"]
            )

    for color, bmax in zip(colors, BMAX_OVER_AH):
        curve = sorted(
            (
                row
                for row in rows
                if finite_float(row["bmax_over_hydrogen_interparticle_spacing"]) == float(bmax)
            ),
            key=lambda row: finite_float(row["resolution_factor"]),
        )
        factors = np.array([finite_float(row["resolution_factor"]) for row in curve], dtype=float)
        drag = np.array([finite_float(row["absolute_drag_N"]) for row in curve], dtype=float)
        reference = reference_by_bmax.get(float(bmax), math.nan)
        relative = np.abs(drag / reference - 1.0) if math.isfinite(reference) and reference != 0.0 else np.nan * drag
        label = rf"$b_{{max}}/a_H={bmax:g}$"

        axes[0].plot(factors, drag, marker="o", linewidth=1.8, markersize=4.0, color=color, label=label)
        axes[1].plot(factors, relative, marker="o", linewidth=1.8, markersize=4.0, color=color, label=label)

    axes[0].set_ylabel("|drag| [N]")
    axes[0].set_yscale("log")
    axes[0].set_title("Absolute drag")

    axes[1].set_ylabel("relative difference from 10x")
    axes[1].set_yscale("log")
    axes[1].set_title("Convergence to highest resolution")

    for axis in axes:
        axis.set_xlabel("resolution scale factor")
        axis.set_xscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)

    first = rows[0]
    fig.suptitle(
        "Dungeon one-case resolution convergence: "
        f"condition {int(first['condition'])}, v={finite_float(first['velocity_cm_s']):.6g} cm/s"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def run_factor_case(
    index: int,
    factor: float,
    condition: int,
    velocity_cm_s: float,
    gpu_count: int,
) -> list[dict[str, object]]:
    resolution = scaled_resolution(factor)
    gpu_id = (index - 1) % gpu_count if gpu_count > 0 else -1
    task = {
        "task_id": index,
        "condition": condition,
        "condition_label": condition_label(condition),
        "velocity_cm_s": velocity_cm_s,
        "gpu_id": gpu_id,
        **resolution,
    }
    print(
        "[convergence start] "
        f"factor={factor:g} condition={condition} velocity_cm_s={velocity_cm_s:.6e} "
        f"vres={resolution['vres']} rhores={resolution['rhores']} "
        f"ures={resolution['ures']} dphires={resolution['dphires']} "
        f"gpu_id={gpu_id if gpu_id >= 0 else 'cpu'}",
        flush=True,
    )
    factor_rows = run_condition_velocity_task(task)
    for row in factor_rows:
        row["resolution_factor"] = factor
        row["reference_condition_label"] = condition_label(condition)
    return factor_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Check one Dungeon case across numerical integration resolutions.")
    parser.add_argument("--condition", type=int, default=3)
    parser.add_argument("--velocity-cm-s", type=float, default=1.0e8)
    parser.add_argument("--factors", default=",".join(f"{factor:g}" for factor in DEFAULT_FACTORS))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("DUNGEON_CPU_CORES", "20")))
    parser.add_argument("--gpus", type=int, default=int(os.environ.get("DUNGEON_GPUS", "2")))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTDIR / "resolution_convergence_case.csv",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=OUTDIR / "resolution_convergence_case.png",
    )
    args = parser.parse_args()

    factors = [float(value) for value in args.factors.split(",") if value.strip()]
    rows: list[dict[str, object]] = []
    worker_count = max(1, min(args.workers, len(factors)))
    gpu_count = max(0, args.gpus)
    print(
        "[convergence run] "
        f"factors={factors} workers={worker_count} gpus={gpu_count} "
        f"condition={args.condition} velocity_cm_s={args.velocity_cm_s:.6e}",
        flush=True,
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(run_factor_case, index, factor, args.condition, args.velocity_cm_s, gpu_count): factor
            for index, factor in enumerate(factors, 1)
        }
        for future in concurrent.futures.as_completed(futures):
            factor = futures[future]
            factor_rows = future.result()
            rows.extend(factor_rows)
        print(
            "[convergence progress] "
            f"finished factor={factor:g} rows={len(factor_rows)}",
            flush=True,
        )

    rows.sort(
        key=lambda row: (
            float(row["bmax_over_hydrogen_interparticle_spacing"]),
            float(row["resolution_factor"]),
        )
    )
    add_shape_columns(rows)
    write_rows_csv(args.output_csv, rows)
    plot_convergence(rows, args.output_png)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
