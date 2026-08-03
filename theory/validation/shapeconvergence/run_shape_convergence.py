# Run from repository root:
# python .\theory\validation\shapeconvergence\run_shape_convergence.py --workers 8
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUTDIR / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CM_PER_S_TO_M_PER_S, condition_label, make_drag, quiet_drag, write_csv

CONDITIONS = (1, 3)
N_VELOCITIES = 16
ACIPC = 1.0
DEFAULT_RHOMAX_FRACTIONS = (0.30, 0.35, 0.40)
DEFAULT_RESOLUTION_SCALES = (0.5, 1.0, 2.0)


def fraction_list(text: str) -> tuple[float, ...]:
    try:
        values = tuple(sorted(set(map(float, text.split(",")))))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated numbers") from exc
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError("provide positive rhomax fractions")
    return values


def condition_list(values: list[int]) -> tuple[int, ...]:
    conditions = tuple(sorted(set(values)))
    unsupported = set(conditions).difference(CONDITIONS)
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"shape convergence is restricted to conditions {CONDITIONS}; got {sorted(unsupported)}"
        )
    return conditions


def bragg_focused_velocities(
    minimum: float,
    bragg_minimum: float,
    bragg_maximum: float,
    maximum: float,
) -> np.ndarray:
    """Return 16 unique points: 3 low-tail, 10 Bragg-region, 3 high-tail."""
    if not 0 < minimum < bragg_minimum < bragg_maximum < maximum:
        raise ValueError("require min < bragg-min < bragg-max < max")
    low = np.geomspace(minimum, bragg_minimum, 4)[:-1]
    bragg = np.geomspace(bragg_minimum, bragg_maximum, 10)
    high = np.geomspace(bragg_maximum, maximum, 4)[1:]
    return np.concatenate((low, bragg, high))


def run_point(task: tuple[int, float, float, float, dict[str, int]]) -> dict[str, object]:
    condition, velocity_cm_s, rhomax_fraction, resolution_scale, resolution = task
    drag = make_drag(
        condition,
        rhomax_fraction=rhomax_fraction,
        acipc=ACIPC,
        **resolution,
    )
    impact_parameter_cutoff_m = drag.rhomax_fraction / drag.ustart
    return {
        "condition": condition,
        "velocity_cm_s": velocity_cm_s,
        "drag_N": quiet_drag(drag, velocity_cm_s * CM_PER_S_TO_M_PER_S),
        "rhomax_fraction": rhomax_fraction,
        "acipc": drag.acipc,
        "impact_parameter_cutoff_m": impact_parameter_cutoff_m,
        "angle_radius_cutoff_m": drag.acipc * impact_parameter_cutoff_m,
        "resolution_scale": resolution_scale,
        "solver": "common.make_drag/DragFourth",
        "finite_start_angle_correction": True,
        **resolution,
    }


def check_zero_force(condition: int, resolution: dict[str, int], tolerance_n: float) -> float:
    drag = make_drag(condition, rhomax_fraction=0.3, acipc=ACIPC, **resolution)
    drag.A = 0.0
    drag.E0Y = 0.0
    force = quiet_drag(drag, 1.0e6 * CM_PER_S_TO_M_PER_S)
    if not np.isfinite(force) or abs(force) > tolerance_n:
        raise RuntimeError(
            f"zero-force invariant failed: drag={force:.6e} N exceeds {tolerance_n:.1e} N"
        )
    return force


def make_plot(
    rows: list[dict[str, object]],
    condition: int,
    fractions: tuple[float, ...],
    resolution_scales: tuple[float, ...],
) -> None:
    fig, axis = plt.subplots(figsize=(9, 6))
    color_maps = {0: plt.cm.Reds, 1: plt.cm.YlOrBr, 2: plt.cm.Greens, 3: plt.cm.Blues}
    colors = color_maps[condition](np.linspace(0.35, 0.9, len(fractions)))
    linestyles = ["--", "-", ":", "-."]
    for color, fraction in zip(colors, fractions):
        for scale_index, scale in enumerate(resolution_scales):
            curve = sorted(
                (
                    row for row in rows
                    if row["condition"] == condition
                    and row["rhomax_fraction"] == fraction
                    and row["resolution_scale"] == scale
                ),
                key=lambda row: float(row["velocity_cm_s"]),
            )
            velocity = np.array([float(row["velocity_cm_s"]) for row in curve])
            drag = np.array([float(row["drag_N"]) for row in curve])
            valid = np.isfinite(drag) & (drag > 0.0)
            axis.plot(
                velocity[valid],
                drag[valid],
                marker="o",
                linestyle=linestyles[scale_index % len(linestyles)],
                color=color,
                linewidth=2.0,
                markersize=4,
                label=rf"$b_{{max}}/a_H={fraction:g}$, resolution $\times${scale:g}",
            )
            invalid = ~valid
            if np.any(invalid):
                axis.scatter(
                    velocity[invalid],
                    np.abs(drag[invalid]),
                    marker="x",
                    s=55,
                    linewidths=1.8,
                    color=color,
                )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("velocity [cm/s]")
    axis.set_ylabel("|drag| [N]")
    axis.set_title(
        f"Condition {condition}: {condition_label(condition)}, "
        f"finite-start correction, acipc={ACIPC:g}"
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"condition_{condition}_rhomax_fraction_shape_convergence.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drag-curve integration convergence using the same solver and finite-start correction as the impact-parameter fit."
    )
    parser.add_argument("--conditions", nargs="+", type=int, choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-velocity-cm-s", type=float, default=3e2)
    parser.add_argument("--max-velocity-cm-s", type=float, default=1e8)
    parser.add_argument("--bragg-min-velocity-cm-s", type=float, default=1e6)
    parser.add_argument("--bragg-max-velocity-cm-s", type=float, default=3e7)
    parser.add_argument("--rhomax-fractions", type=fraction_list, default=DEFAULT_RHOMAX_FRACTIONS)
    parser.add_argument(
        "--resolution-scales",
        type=fraction_list,
        default=DEFAULT_RESOLUTION_SCALES,
        help="Comma-separated multipliers applied together to vres, rhores, ures, and dphires.",
    )
    parser.add_argument("--vres", type=int, default=201)
    parser.add_argument("--rhores", type=int, default=180)
    parser.add_argument("--ures", type=int, default=180)
    parser.add_argument("--dphires", type=int, default=180)
    parser.add_argument("--zero-force-tolerance-N", type=float, default=1e-30)
    args = parser.parse_args()
    if args.workers < 1 or args.min_velocity_cm_s <= 0 or args.max_velocity_cm_s <= args.min_velocity_cm_s:
        parser.error("workers and velocity bounds must be positive")

    try:
        velocities = bragg_focused_velocities(
            args.min_velocity_cm_s,
            args.bragg_min_velocity_cm_s,
            args.bragg_max_velocity_cm_s,
            args.max_velocity_cm_s,
        )
    except ValueError as exc:
        parser.error(str(exc))
    conditions = condition_list(args.conditions)
    base_resolution = {name: getattr(args, name) for name in ("vres", "rhores", "ures", "dphires")}
    resolutions = {
        scale: {name: max(3, int(round(value * scale))) for name, value in base_resolution.items()}
        for scale in args.resolution_scales
    }
    for condition in conditions:
        for scale, resolution in resolutions.items():
            zero_force = check_zero_force(condition, resolution, args.zero_force_tolerance_N)
            print(
                f"Condition {condition}, resolution x{scale:g}: "
                f"zero-force invariant passed, drag={zero_force:.3e} N.",
                flush=True,
            )
    tasks = [
        (condition, float(velocity), fraction, scale, resolution)
        for condition in conditions
        for fraction in args.rhomax_fractions
        for scale, resolution in resolutions.items()
        for velocity in velocities
    ]
    print(
        f"Conditions {conditions}: 16 Bragg-focused velocities x "
        f"{len(args.rhomax_fractions)} bmax/aH values x {len(resolutions)} resolution levels "
        f"= {len(tasks)} jobs "
        f"on up to {args.workers} processes; acipc={ACIPC:g}.",
        flush=True,
    )
    start = time.perf_counter()
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_point, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            elapsed = time.perf_counter() - start
            eta = elapsed * (len(tasks) - completed) / completed
            print(f"[{completed:2d}/{len(tasks)}] elapsed={elapsed/60:.1f} min eta={eta/60:.1f} min", flush=True)

    rows.sort(
        key=lambda row: (
            int(row["condition"]),
            float(row["rhomax_fraction"]),
            float(row["resolution_scale"]),
            float(row["velocity_cm_s"]),
        )
    )
    reference_scale = max(args.resolution_scales)
    reference_drag = {
        (int(row["condition"]), float(row["rhomax_fraction"]), float(row["velocity_cm_s"])): float(row["drag_N"])
        for row in rows
        if float(row["resolution_scale"]) == reference_scale
    }
    for row in rows:
        key = (int(row["condition"]), float(row["rhomax_fraction"]), float(row["velocity_cm_s"]))
        value = float(row["drag_N"])
        reference = reference_drag.get(key, np.nan)
        row["reference_resolution_scale"] = reference_scale
        row["relative_error_vs_highest_resolution"] = (
            abs(value - reference) / abs(reference)
            if np.isfinite(value) and np.isfinite(reference) and reference != 0.0
            else np.nan
        )
        row["status"] = "ok" if np.isfinite(value) and value > 0.0 else "invalid_drag"
    write_csv(OUTDIR / "rhomax_fraction_shape_convergence.csv", rows)
    for condition in conditions:
        make_plot(rows, condition, args.rhomax_fractions, args.resolution_scales)
    print(f"Finished in {(time.perf_counter()-start)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    freeze_support()
    main()
