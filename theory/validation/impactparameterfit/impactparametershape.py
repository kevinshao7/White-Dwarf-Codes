# Run from repository root:
# python .\theory\validation\impactparameterfit\impactparametershape.py --workers 8
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUTDIR / ".matplotlib"))


CONDITIONS = (1, 3)
CM_PER_S_TO_M_PER_S = 1.0e-2
DEFAULT_CUTOFF_RADIUS_FACTOR = 50.0
DEFAULT_BMAX_OVER_SPACING = (0.5, 1.0, 2.0, 4.0)

from resolution_scaling import scaled_resolution_for_bmax


def positive_float_list(text: str) -> tuple[float, ...]:
    try:
        values = tuple(sorted(set(float(value) for value in text.split(","))))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated numbers") from exc
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("provide positive values")
    return values


def condition_list(values: list[int]) -> tuple[int, ...]:
    conditions = tuple(sorted(set(values)))
    unsupported = set(conditions).difference(CONDITIONS)
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"impact-parameter shape comparison is restricted to conditions {CONDITIONS}; "
            f"got {sorted(unsupported)}"
        )
    return conditions


def log_velocity_grid(minimum_cm_s: float, maximum_cm_s: float, points: int) -> list[float]:
    if minimum_cm_s <= 0.0 or maximum_cm_s <= minimum_cm_s:
        raise ValueError("velocity bounds must satisfy 0 < min < max")
    if points < 3:
        raise ValueError("curve-points must be at least 3")
    step = (math.log(maximum_cm_s) - math.log(minimum_cm_s)) / (points - 1)
    return [math.exp(math.log(minimum_cm_s) + index * step) for index in range(points)]


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_curve_point(task: tuple[int, float, float, dict[str, int]]) -> dict[str, object]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import condition_label, make_drag, quiet_drag

    condition, velocity_cm_s, bmax_over_spacing, resolution = task
    rhomax_fraction = bmax_over_spacing / DEFAULT_CUTOFF_RADIUS_FACTOR
    if rhomax_fraction > 1.0:
        raise ValueError(
            f"bmax/aH={bmax_over_spacing:g} exceeds finite launch radius "
            f"{DEFAULT_CUTOFF_RADIUS_FACTOR:g} aH"
        )

    scaled_resolution = scaled_resolution_for_bmax(resolution, bmax_over_spacing)
    drag = make_drag(
        condition,
        rhomax_fraction=rhomax_fraction,
        cutoff_radius_factor=DEFAULT_CUTOFF_RADIUS_FACTOR,
        **scaled_resolution,
    )
    force_n = quiet_drag(drag, velocity_cm_s * CM_PER_S_TO_M_PER_S)
    acceleration_cm_s2 = abs(force_n / drag.ms) * 100.0
    hydrogen_interparticle_spacing_m = 1.0 / (DEFAULT_CUTOFF_RADIUS_FACTOR * drag.ustart)
    finite_radius_m = 1.0 / drag.ustart
    impact_parameter_cutoff_m = bmax_over_spacing * hydrogen_interparticle_spacing_m

    return {
        "condition": condition,
        "condition_label": condition_label(condition),
        "velocity_cm_s": velocity_cm_s,
        "velocity_m_s": velocity_cm_s * CM_PER_S_TO_M_PER_S,
        "bmax_over_hydrogen_interparticle_spacing": bmax_over_spacing,
        "rhomax_fraction_of_naive_outer_radius": rhomax_fraction,
        "cutoff_radius_factor": DEFAULT_CUTOFF_RADIUS_FACTOR,
        "hydrogen_interparticle_spacing_m": hydrogen_interparticle_spacing_m,
        "impact_parameter_cutoff_m": impact_parameter_cutoff_m,
        "finite_radius_m": finite_radius_m,
        "angular_momentum_geometry": "naive finite radius with DragFourth: L=mu*b*v_inf",
        "drag_N": force_n,
        "absolute_drag_N": abs(force_n),
        "model_acceleration_cm_s2": acceleration_cm_s2,
        "status": "ok" if math.isfinite(force_n) and force_n != 0.0 else "invalid_drag",
        "base_rhores_at_bmax_over_aH_1": resolution["rhores"],
        "base_dphires_at_bmax_over_aH_1": resolution["dphires"],
        **scaled_resolution,
    }


def add_shape_columns(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[int, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["condition"]), float(row["bmax_over_hydrogen_interparticle_spacing"]))
        groups.setdefault(key, []).append(row)

    for group in groups.values():
        group.sort(key=lambda row: float(row["velocity_cm_s"]))
        drag = [float(row["absolute_drag_N"]) for row in group]
        velocity = [float(row["velocity_cm_s"]) for row in group]
        valid_indices = [
            index for index, value in enumerate(drag)
            if math.isfinite(value) and value > 0.0 and math.isfinite(velocity[index]) and velocity[index] > 0.0
        ]
        peak_drag = max((drag[index] for index in valid_indices), default=math.nan)
        slope = [math.nan] * len(group)
        if len(valid_indices) >= 3:
            for valid_position, index in enumerate(valid_indices):
                if valid_position == 0:
                    left = valid_indices[valid_position]
                    right = valid_indices[valid_position + 1]
                elif valid_position == len(valid_indices) - 1:
                    left = valid_indices[valid_position - 1]
                    right = valid_indices[valid_position]
                else:
                    left = valid_indices[valid_position - 1]
                    right = valid_indices[valid_position + 1]
                slope[index] = (math.log(drag[right]) - math.log(drag[left])) / (
                    math.log(velocity[right]) - math.log(velocity[left])
                )
        for index, row in enumerate(group):
            value = float(row["absolute_drag_N"])
            row["peak_drag_N_for_condition_and_bmax"] = peak_drag
            row["drag_normalized_to_peak"] = value / peak_drag if math.isfinite(peak_drag) and peak_drag > 0.0 else math.nan
            row["local_log_slope_dlogdrag_dlogv"] = slope[index]


def plot_condition(rows: list[dict[str, object]], condition: int, bmax_values: tuple[float, ...], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import condition_label

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(bmax_values)))

    for color, bmax in zip(colors, bmax_values):
        curve = sorted(
            (
                row
                for row in rows
                if int(row["condition"]) == condition
                and float(row["bmax_over_hydrogen_interparticle_spacing"]) == bmax
            ),
            key=lambda row: float(row["velocity_cm_s"]),
        )
        velocity = np.array([float(row["velocity_cm_s"]) for row in curve], dtype=float)
        drag = np.array([float(row["absolute_drag_N"]) for row in curve], dtype=float)
        normalized = np.array([float(row["drag_normalized_to_peak"]) for row in curve], dtype=float)
        valid = np.isfinite(drag) & (drag > 0.0) & np.isfinite(normalized) & (normalized > 0.0)
        label = rf"$b_{{max}}/a_H={bmax:g}$"

        axes[0].plot(velocity[valid], drag[valid], marker="o", linewidth=2.0, markersize=4, color=color, label=label)
        axes[1].plot(
            velocity[valid],
            normalized[valid],
            marker="o",
            linewidth=2.0,
            markersize=4,
            color=color,
            label=label,
        )

    axes[0].set_yscale("log")
    axes[0].set_ylabel("|drag| [N]")
    axes[0].set_title("Absolute drag")

    axes[1].set_yscale("log")
    axes[1].set_ylabel("drag normalized to each curve peak")
    axes[1].set_title("Curve shape")

    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("velocity [cm/s]")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)

    fig.suptitle(
        f"Condition {condition}: {condition_label(condition)}, "
        rf"impact-parameter cutoff sweep; launch radius={DEFAULT_CUTOFF_RADIUS_FACTOR:g} $a_H$"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot drag-curve shape changes for fixed choices of impact-parameter cutoff bmax/aH."
    )
    parser.add_argument("--conditions", nargs="+", type=int, choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bmax-over-spacing", type=positive_float_list, default=DEFAULT_BMAX_OVER_SPACING)
    parser.add_argument("--min-velocity-cm-s", type=float, default=3.0e2)
    parser.add_argument("--max-velocity-cm-s", type=float, default=1.0e8)
    parser.add_argument("--curve-points", type=int, default=24)
    parser.add_argument("--vres", type=int, default=201)
    parser.add_argument("--rhores", type=int, default=180)
    parser.add_argument("--ures", type=int, default=180)
    parser.add_argument("--dphires", type=int, default=180)
    parser.add_argument("--output-csv", type=Path, default=OUTDIR / "impact_parameter_shape_curves.csv")
    parser.add_argument(
        "--output-png-template",
        type=str,
        default=str(OUTDIR / "condition_{condition}_impact_parameter_shape.png"),
        help="Output PNG path template. Use {condition} for the condition number.",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.rhores < 1:
        parser.error("--rhores is the bin count at bmax/aH=1 and must be positive")
    if args.dphires < 1:
        parser.error("--dphires is the scattering-angle bin count at bmax/aH=1 and must be positive")
    if max(args.bmax_over_spacing) > DEFAULT_CUTOFF_RADIUS_FACTOR:
        parser.error(
            f"--bmax-over-spacing values must not exceed the finite launch radius "
            f"{DEFAULT_CUTOFF_RADIUS_FACTOR:g} a_H"
        )

    try:
        velocities = log_velocity_grid(args.min_velocity_cm_s, args.max_velocity_cm_s, args.curve_points)
    except ValueError as exc:
        parser.error(str(exc))

    conditions = condition_list(args.conditions)
    resolution = {name: getattr(args, name) for name in ("vres", "rhores", "ures", "dphires")}
    tasks = [
        (condition, float(velocity), bmax_over_spacing, resolution)
        for condition in conditions
        for bmax_over_spacing in args.bmax_over_spacing
        for velocity in velocities
    ]

    print(
        f"Conditions {conditions}: {len(velocities)} velocities x "
        f"{len(args.bmax_over_spacing)} bmax/aH values = {len(tasks)} jobs "
        f"on up to {args.workers} processes.",
        flush=True,
    )

    start = time.perf_counter()
    rows: list[dict[str, object]] = []
    if args.workers == 1:
        for completed, task in enumerate(tasks, 1):
            rows.append(run_curve_point(task))
            elapsed = time.perf_counter() - start
            eta = elapsed * (len(tasks) - completed) / completed
            print(f"[{completed:2d}/{len(tasks)}] elapsed={elapsed/60:.1f} min eta={eta/60:.1f} min", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_curve_point, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), 1):
                rows.append(future.result())
                elapsed = time.perf_counter() - start
                eta = elapsed * (len(tasks) - completed) / completed
                print(f"[{completed:2d}/{len(tasks)}] elapsed={elapsed/60:.1f} min eta={eta/60:.1f} min", flush=True)

    rows.sort(
        key=lambda row: (
            int(row["condition"]),
            float(row["bmax_over_hydrogen_interparticle_spacing"]),
            float(row["velocity_cm_s"]),
        )
    )
    add_shape_columns(rows)
    write_rows_csv(args.output_csv, rows)
    print(f"Wrote {args.output_csv}", flush=True)

    for condition in conditions:
        output_png = Path(args.output_png_template.format(condition=condition))
        plot_condition(rows, condition, args.bmax_over_spacing, output_png)
        print(f"Wrote {output_png}", flush=True)

    print(f"Finished in {(time.perf_counter() - start)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    freeze_support()
    main()
