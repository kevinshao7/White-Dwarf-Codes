# Run from repository root:
#   python .\theory\finite\convergence\run_bmax_convergence.py --workers 1
"""Convergence of finite-launch drag as ``b_max = r_i`` grows toward infinity.

`FiniteLaunchDrag` ties the impact-parameter ceiling to the launch radius:
``b_max == r_i == rhomax_fraction * a_H``. This script sweeps a finite set of
``b_max/a_H`` values and checks how quickly the predicted drag force
approaches its large-``b_max`` limit across bulk velocity, for all four
conditions.

The practical reference is the largest tested value in ``--bmax-over-ah``.
That is not literal infinity; it is the finite curve the smaller ``b_max``
choices are compared against.

To keep the quadrature accuracy from drifting as the launch sphere expands,
the total integration-point count is multiplied by 1.5 for every decade
increase in ``b_max/a_H`` relative to a reference value (default `0.1`).

For the impact-parameter integral, this script keeps the inner cutoff fixed at
``b_min = 1e-6 * a_H`` and preserves a constant point density in ``ln b``.
That means ``rhores`` grows in proportion to ``ln(b_max / b_min)`` rather than
staying fixed or scaling by an arbitrary power law.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUTDIR / ".matplotlib"))

import numpy as np

THEORY_DIR = Path(__file__).resolve().parents[2]
if str(THEORY_DIR) not in sys.path:
    sys.path.insert(0, str(THEORY_DIR))

from finite.finite_launch import FiniteLaunchDrag  # noqa: E402
from finite.progress import run_pool_with_heartbeat  # noqa: E402

ALL_CONDITIONS = (0, 1, 2, 3)
DEFAULT_BMAX_OVER_AH = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
DEFAULT_N_VELOCITIES = 16
DEFAULT_VELOCITY_MIN_CM_S = 1.0e5
DEFAULT_VELOCITY_MAX_CM_S = 1.0e8
DEFAULT_VRES = 101
DEFAULT_RHORES = 360
DEFAULT_DPHIRES = 360
DEFAULT_TOTAL_RESOLUTION_SCALE_PER_DECADE = 1.5
DEFAULT_RESOLUTION_SCALE_REFERENCE_BMAX_OVER_AH = 0.1
DEFAULT_DRAG_BATCH_CHUNK_SIZE = 1
DEFAULT_FIXED_BMIN_OVER_AH = 1.0e-6
CM_PER_M = 100.0
CONDITION_SUBPLOT_POSITION = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scaled_resolution(
    base_resolution: int,
    bmax_over_aH: float,
    total_scale_per_decade: float,
    reference_bmax_over_aH: float,
    axis_count: int = 3,
) -> int:
    decades = math.log10(bmax_over_aH / reference_bmax_over_aH)
    per_axis_scale = (total_scale_per_decade**decades) ** (1.0 / axis_count)
    scaled = base_resolution * per_axis_scale
    return max(1, int(math.ceil(scaled)))


def scaled_rhores(
    base_rhores: int,
    bmax_over_aH: float,
    fixed_bmin_over_aH: float,
    reference_bmax_over_aH: float,
) -> int:
    reference_span = math.log(reference_bmax_over_aH / fixed_bmin_over_aH)
    current_span = math.log(bmax_over_aH / fixed_bmin_over_aH)
    density = base_rhores / reference_span
    return max(1, int(math.ceil(density * current_span)))


def compute_curve_task(task: tuple[int, float, np.ndarray, int, int, int, float, float, float, int]) -> list[dict[str, object]]:
    (
        condition,
        bmax_over_aH,
        velocities_cm_s,
        base_vres,
        base_rhores,
        base_dphires,
        total_scale_per_decade,
        reference_bmax_over_aH,
        fixed_bmin_over_aH,
        drag_batch_chunk_size,
    ) = task
    rhores = scaled_rhores(base_rhores, bmax_over_aH, fixed_bmin_over_aH, reference_bmax_over_aH)
    rhores_scale_factor = rhores / base_rhores
    resolution_scale_factor = total_scale_per_decade ** math.log10(bmax_over_aH / reference_bmax_over_aH)
    remaining_scale_factor = resolution_scale_factor / rhores_scale_factor
    remaining_scale_factor = max(remaining_scale_factor, 1.0 / max(base_vres, base_dphires, 1))
    per_axis_scale_factor = remaining_scale_factor ** 0.5
    vres = max(1, int(math.ceil(base_vres * per_axis_scale_factor)))
    dphires = max(1, int(math.ceil(base_dphires * per_axis_scale_factor)))

    drag = FiniteLaunchDrag(
        condition,
        method="vectorized",
        rhomax_fraction=bmax_over_aH,
        vres=vres,
        rhores=rhores,
        dphires=dphires,
        bmin_fraction=fixed_bmin_over_aH / bmax_over_aH,
    )
    velocities_m_s = np.asarray(velocities_cm_s, dtype=np.float64) / CM_PER_M

    force_chunks: list[np.ndarray] = []
    for start in range(0, len(velocities_m_s), drag_batch_chunk_size):
        stop = min(start + drag_batch_chunk_size, len(velocities_m_s))
        vb = velocities_m_s[start:stop]
        if drag_batch_chunk_size == 1:
            force_chunks.append(np.array([float(drag.drag(float(v))) for v in vb], dtype=np.float64))
        else:
            force_chunks.append(np.asarray(drag.drag_batch(vb), dtype=np.float64))
    forces_n = np.concatenate(force_chunks) if force_chunks else np.empty(0, dtype=np.float64)

    rows: list[dict[str, object]] = []
    for velocity_index, (velocity_cm_s, velocity_m_s, force_n) in enumerate(
        zip(velocities_cm_s, velocities_m_s, forces_n, strict=True)
    ):
        rows.append(
            {
                "condition": condition,
                "bmax_over_aH": bmax_over_aH,
                "velocity_index": velocity_index,
                "velocity_cm_s": float(velocity_cm_s),
                "velocity_m_s": float(velocity_m_s),
                "force_n": float(force_n),
                "base_vres": base_vres,
                "base_rhores": base_rhores,
                "base_dphires": base_dphires,
                "vres": vres,
                "rhores": rhores,
                "dphires": dphires,
                "resolution_scale_factor": resolution_scale_factor,
                "resolution_scale_per_decade": total_scale_per_decade,
                "resolution_scale_per_axis_factor": per_axis_scale_factor,
                "fixed_bmin_over_aH": fixed_bmin_over_aH,
                "bmin_fraction": fixed_bmin_over_aH / bmax_over_aH,
                "resolution_scale_reference_bmax_over_aH": reference_bmax_over_aH,
                "drag_batch_chunk_size": drag_batch_chunk_size,
                "temperature_K": float(drag.T),
                "density_g_cc": float(drag.gcc),
            }
        )
    return rows


def add_relative_error(rows: list[dict[str, object]], reference_bmax_over_aH: float) -> None:
    reference_force = {
        (row["condition"], row["velocity_index"]): row["force_n"]
        for row in rows
        if math.isclose(float(row["bmax_over_aH"]), reference_bmax_over_aH, rel_tol=1e-12)
    }
    for row in rows:
        ref = float(reference_force[(row["condition"], row["velocity_index"])])
        row["relative_error_vs_largest_bmax"] = abs(float(row["force_n"]) - ref) / abs(ref) if ref != 0.0 else math.nan


def run_tasks_sequentially(tasks: list[tuple[object, ...]]) -> list[list[dict[str, object]]]:
    results: list[list[dict[str, object]]] = []
    total = len(tasks)
    start = time.perf_counter()
    for index, task in enumerate(tasks, start=1):
        results.append(compute_curve_task(task))
        elapsed = time.perf_counter() - start
        eta = elapsed * (total - index) / index if index else math.nan
        print(f"[bmax_convergence] {index}/{total} done  elapsed={elapsed:.1f}s  eta={eta:.1f}s  in_flight=0", flush=True)
    return results


def _bmax_styles(bmax_values: tuple[float, ...]) -> tuple[dict[float, str], dict[float, tuple[float, float, float, float]]]:
    import matplotlib.pyplot as plt

    ordered = sorted(bmax_values)
    linestyles = {bmax: "solid" for bmax in ordered}
    cmap = plt.get_cmap("rainbow")
    if len(ordered) == 1:
        colors = {ordered[0]: cmap(0.8)}
    else:
        colors = {bmax: cmap(i / (len(ordered) - 1)) for i, bmax in enumerate(ordered)}
    return linestyles, colors


def _condition_label(condition: int) -> str:
    drag = FiniteLaunchDrag(condition, method="vectorized")
    return f"Condition {condition}\nT = {drag.T:.0f} K, density = {drag.gcc:.1e} g/cc"


def make_force_plot(rows: list[dict[str, object]], bmax_values: tuple[float, ...]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), sharex=True, sharey=True)
    linestyles, colors = _bmax_styles(bmax_values)

    for condition, (row, col) in CONDITION_SUBPLOT_POSITION.items():
        axis = axes[row, col]
        for bmax_over_aH in sorted(bmax_values):
            curve = sorted(
                (
                    item
                    for item in rows
                    if item["condition"] == condition
                    and math.isclose(float(item["bmax_over_aH"]), bmax_over_aH, rel_tol=1e-12)
                ),
                key=lambda item: float(item["velocity_cm_s"]),
            )
            velocity_cm_s = np.array([float(item["velocity_cm_s"]) for item in curve], dtype=np.float64)
            force_n = np.array([float(item["force_n"]) for item in curve], dtype=np.float64)
            valid = np.isfinite(force_n) & (force_n > 0.0)
            axis.plot(
                velocity_cm_s[valid],
                force_n[valid],
                color=colors[bmax_over_aH],
                linestyle=linestyles[bmax_over_aH],
                linewidth=2.0,
                label=fr"$b_{{max}}/a_H={bmax_over_aH:g}$",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(_condition_label(condition), fontsize=10)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8, loc="best")

    for axis in axes[1, :]:
        axis.set_xlabel("bulk velocity $v_b$ (cm/s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("drag force $F$ (N)")

    fig.suptitle(r"$b_{max}=r_i$ convergence toward the large-$b_{max}$ limit", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(OUTDIR / "bmax_convergence_force_vs_velocity.png", dpi=200)
    plt.close(fig)


def make_relative_error_plot(rows: list[dict[str, object]], bmax_values: tuple[float, ...], reference_bmax_over_aH: float) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), sharex=True, sharey=True)
    linestyles, colors = _bmax_styles(bmax_values)

    for condition, (row, col) in CONDITION_SUBPLOT_POSITION.items():
        axis = axes[row, col]
        for bmax_over_aH in sorted(bmax_values):
            if math.isclose(bmax_over_aH, reference_bmax_over_aH, rel_tol=1e-12):
                continue
            curve = sorted(
                (
                    item
                    for item in rows
                    if item["condition"] == condition
                    and math.isclose(float(item["bmax_over_aH"]), bmax_over_aH, rel_tol=1e-12)
                ),
                key=lambda item: float(item["velocity_cm_s"]),
            )
            velocity_cm_s = np.array([float(item["velocity_cm_s"]) for item in curve], dtype=np.float64)
            relerr = np.array([float(item["relative_error_vs_largest_bmax"]) for item in curve], dtype=np.float64)
            valid = np.isfinite(relerr) & (relerr > 0.0)
            axis.plot(
                velocity_cm_s[valid],
                relerr[valid],
                color=colors[bmax_over_aH],
                linestyle=linestyles[bmax_over_aH],
                linewidth=2.0,
                label=fr"$b_{{max}}/a_H={bmax_over_aH:g}$",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(_condition_label(condition), fontsize=10)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8, loc="best")

    for axis in axes[1, :]:
        axis.set_xlabel("bulk velocity $v_b$ (cm/s)")
    for axis in axes[:, 0]:
        axis.set_ylabel(fr"relative error vs. $b_{{max}}/a_H={reference_bmax_over_aH:g}$")

    fig.suptitle(r"$b_{max}=r_i$ convergence error relative to the largest tested $b_{max}$", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(OUTDIR / "bmax_convergence_relative_error_vs_velocity.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", type=int, default=list(ALL_CONDITIONS))
    parser.add_argument("--bmax-over-ah", nargs="+", type=float, default=list(DEFAULT_BMAX_OVER_AH))
    parser.add_argument("--n-velocities", type=int, default=DEFAULT_N_VELOCITIES)
    parser.add_argument("--velocity-min-cm-s", type=float, default=DEFAULT_VELOCITY_MIN_CM_S)
    parser.add_argument("--velocity-max-cm-s", type=float, default=DEFAULT_VELOCITY_MAX_CM_S)
    parser.add_argument("--vres", type=int, default=DEFAULT_VRES)
    parser.add_argument("--rhores", type=int, default=DEFAULT_RHORES)
    parser.add_argument("--dphires", type=int, default=DEFAULT_DPHIRES)
    parser.add_argument(
        "--resolution-scale-per-decade",
        type=float,
        default=DEFAULT_TOTAL_RESOLUTION_SCALE_PER_DECADE,
        help="Multiply the total integration-point count by this factor for each decade increase in b_max/a_H while keeping the default log-b grid density fixed.",
    )
    parser.add_argument(
        "--resolution-scale-reference-bmax-over-ah",
        type=float,
        default=DEFAULT_RESOLUTION_SCALE_REFERENCE_BMAX_OVER_AH,
        help="b_max/a_H value at which the base --vres, --rhores, and --dphires define the reference integration-point budget.",
    )
    parser.add_argument(
        "--fixed-bmin-over-ah",
        type=float,
        default=DEFAULT_FIXED_BMIN_OVER_AH,
        help="Absolute inner cutoff b_min expressed in units of a_H; this stays fixed while b_max changes.",
    )
    parser.add_argument(
        "--drag-batch-chunk-size",
        type=int,
        default=DEFAULT_DRAG_BATCH_CHUNK_SIZE,
        help="How many velocities to pass to drag_batch at once inside one task. Use 1 for the safest path.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=12.0,
        help="Print a status line at least this often even if no task has finished yet.",
    )
    args = parser.parse_args()

    conditions = tuple(sorted(set(args.conditions)))
    bmax_values = tuple(sorted(set(float(value) for value in args.bmax_over_ah)))
    reference_bmax_over_aH = max(bmax_values)
    if args.resolution_scale_per_decade <= 0.0:
        parser.error("--resolution-scale-per-decade must be positive.")
    if args.resolution_scale_reference_bmax_over_ah <= 0.0:
        parser.error("--resolution-scale-reference-bmax-over-ah must be positive.")
    if args.fixed_bmin_over_ah <= 0.0:
        parser.error("--fixed-bmin-over-ah must be positive.")
    if args.drag_batch_chunk_size < 1:
        parser.error("--drag-batch-chunk-size must be at least 1.")
    if any(value <= 0.0 for value in bmax_values):
        parser.error("--bmax-over-ah values must all be positive.")
    if args.fixed_bmin_over_ah >= min(bmax_values):
        parser.error("--fixed-bmin-over-ah must be smaller than every --bmax-over-ah value.")
    if args.fixed_bmin_over_ah >= args.resolution_scale_reference_bmax_over_ah:
        parser.error("--fixed-bmin-over-ah must be smaller than --resolution-scale-reference-bmax-over-ah.")

    velocities_cm_s = np.logspace(
        math.log10(args.velocity_min_cm_s),
        math.log10(args.velocity_max_cm_s),
        args.n_velocities,
    )

    tasks = [
        (
            condition,
            bmax_over_aH,
            velocities_cm_s,
            args.vres,
            args.rhores,
            args.dphires,
            args.resolution_scale_per_decade,
            args.resolution_scale_reference_bmax_over_ah,
            args.fixed_bmin_over_ah,
            args.drag_batch_chunk_size,
        )
        for condition in conditions
        for bmax_over_aH in bmax_values
    ]

    start = time.perf_counter()
    print(
        f"Running {len(tasks)} drag evaluations across conditions {list(conditions)} "
        f"for b_max/a_H values {list(bmax_values)} with {args.n_velocities} velocities "
        f"on up to {args.workers} workers. Base resolutions "
        f"(base vres={args.vres}, rhores={args.rhores}, dphires={args.dphires}) use fixed "
        f"b_min/a_H={args.fixed_bmin_over_ah:g} and constant log-b density, while the total "
        f"integration-point count grows by {args.resolution_scale_per_decade:g}x per decade from "
        f"b_max/a_H={args.resolution_scale_reference_bmax_over_ah:g}; "
        f"drag_batch chunk size = {args.drag_batch_chunk_size}.",
        flush=True,
    )

    if args.workers <= 1:
        curve_rows = run_tasks_sequentially(tasks)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            curve_rows = run_pool_with_heartbeat(
                pool, tasks, compute_curve_task, heartbeat_seconds=args.heartbeat_seconds, label="bmax_convergence"
            )

    rows = [row for chunk in curve_rows for row in chunk]
    rows.sort(key=lambda row: (row["condition"], row["bmax_over_aH"], row["velocity_index"]))
    add_relative_error(rows, reference_bmax_over_aH)
    write_csv(OUTDIR / "bmax_convergence_scan.csv", rows)
    make_force_plot(rows, bmax_values)
    make_relative_error_plot(rows, bmax_values, reference_bmax_over_aH)

    print(
        "Wrote bmax_convergence_force_vs_velocity.png, "
        "bmax_convergence_relative_error_vs_velocity.png, and "
        f"bmax_convergence_scan.csv to {OUTDIR}.",
        flush=True,
    )
    print(f"Finished in {(time.perf_counter() - start):.1f}s.", flush=True)


if __name__ == "__main__":
    freeze_support()
    main()
