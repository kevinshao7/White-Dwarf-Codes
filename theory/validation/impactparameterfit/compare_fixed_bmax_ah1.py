# Run from repository root: python .\theory\validation\impactparameterfit\compare_fixed_bmax_ah1.py --workers 8
from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fit_bmax_to_lammps import (
    ALL_CONDITIONS,
    COUPLING_PARAMETER,
    REPO_ROOT,
    condition_curve_velocities,
    condition_label,
    filter_points,
    load_lammps_expfit_points,
    load_points_from_csv,
    run_curve_case,
    run_fit_point_case,
    select_fit_points,
    write_csv,
    FIT_PARAMETER,
)


OUTDIR = Path(__file__).resolve().parent
FIXED_BMAX_OVER_AH = 1.0


def point_diagnostic(task: tuple[int, float, str, object, int, int, int, int]) -> dict[str, object]:
    row = run_fit_point_case(task)
    data = float(row["data_acceleration_cm_s2"])
    model = float(row["model_acceleration_cm_s2"])
    row["fractional_residual_model_minus_data_over_data"] = (model - data) / data
    row["absolute_fractional_residual"] = abs(model - data) / data
    return row


def summarize(condition: int, rows: list[dict[str, object]]) -> dict[str, object]:
    fractional = np.array(
        [float(row["fractional_residual_model_minus_data_over_data"]) for row in rows], dtype=float
    )
    log_residual = np.array([float(row["log_residual"]) for row in rows], dtype=float)
    weighted = np.array([float(row["weighted_log_residual"]) for row in rows], dtype=float)
    return {
        "condition": condition,
        "condition_label": condition_label(condition),
        "coupling_parameter": COUPLING_PARAMETER.get(condition, math.nan),
        "bmax_over_aH": FIXED_BMAX_OVER_AH,
        "n_points": len(rows),
        "mean_fractional_residual": float(np.mean(fractional)),
        "mean_absolute_fractional_residual": float(np.mean(np.abs(fractional))),
        "rms_fractional_residual": float(np.sqrt(np.mean(np.square(fractional)))),
        "maximum_absolute_fractional_residual": float(np.max(np.abs(fractional))),
        "rmse_log": float(np.sqrt(np.mean(np.square(log_residual)))),
        "mean_squared_weighted_log_residual": float(np.mean(np.square(weighted))),
    }


def plot_comparison(all_points, selected_points, curve_rows, summaries, conditions, output: Path) -> None:
    colors = {0: "tab:red", 1: "tab:orange", 2: "tab:green", 3: "tab:blue"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=False, sharey=False)

    for ax, condition in zip(axes.flat, conditions):
        color = colors[condition]
        data = [point for point in all_points if point.condition == condition]
        selected = [point for point in selected_points if point.condition == condition]
        curve = sorted(
            (row for row in curve_rows if int(row["condition"]) == condition and row["status"] == "ok"),
            key=lambda row: float(row["velocity_cm_s"]),
        )
        summary = next(row for row in summaries if int(row["condition"]) == condition)

        ax.errorbar(
            [point.velocity_cm_s for point in data],
            [point.acceleration_cm_s2 for point in data],
            xerr=[point.velocity_sigma_cm_s if np.isfinite(point.velocity_sigma_cm_s) else 0.0 for point in data],
            yerr=[point.acceleration_sigma_cm_s2 if np.isfinite(point.acceleration_sigma_cm_s2) else 0.0 for point in data],
            fmt="o",
            markersize=3,
            alpha=0.25,
            color=color,
            label="data",
        )
        ax.scatter(
            [point.velocity_cm_s for point in selected],
            [point.acceleration_cm_s2 for point in selected],
            s=45,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            label="comparison points",
            zorder=3,
        )
        ax.plot(
            [float(row["velocity_cm_s"]) for row in curve],
            [float(row["model_acceleration_cm_s2"]) for row in curve],
            color="black",
            linewidth=2,
            label=r"model: $b_{\max}/a_H=1$",
        )
        ax.text(
            0.04,
            0.05,
            f"mean |fractional error| = {float(summary['mean_absolute_fractional_residual']):.3g}\n"
            f"RMS fractional error = {float(summary['rms_fractional_residual']):.3g}\n"
            f"max |fractional error| = {float(summary['maximum_absolute_fractional_residual']):.3g}",
            transform=ax.transAxes,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
        ax.set_title(f"Condition {condition}: {condition_label(condition)}, Gamma={COUPLING_PARAMETER[condition]:.2g}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("v (cm/s)")
        ax.set_ylabel("a (cm/s^2)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(conditions):]:
        ax.set_visible(False)

    fig.suptitle(r"Single fixed impact parameter for all conditions: $b_{\max}/a_H=1$")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_shape_diagnostics(prediction_rows, curve_rows, conditions, output: Path) -> None:
    colors = {0: "tab:red", 1: "tab:orange", 2: "tab:green", 3: "tab:blue"}
    fig, axes = plt.subplots(3, len(conditions), figsize=(5.2 * len(conditions), 12), squeeze=False)
    for column, condition in enumerate(conditions):
        color = colors[condition]
        curve = sorted(
            (
                row for row in curve_rows
                if int(row["condition"]) == condition and row["status"] == "ok"
            ),
            key=lambda row: float(row["velocity_cm_s"]),
        )
        points = [row for row in prediction_rows if int(row["condition"]) == condition]
        velocity = np.array([float(row["velocity_cm_s"]) for row in curve])
        acceleration = np.array([float(row["model_acceleration_cm_s2"]) for row in curve])
        slope = np.gradient(np.log(acceleration), np.log(velocity))

        shape_ax = axes[0, column]
        shape_ax.plot(velocity, acceleration, color=color, linewidth=2)
        shape_ax.scatter(
            [float(row["velocity_cm_s"]) for row in points],
            [float(row["data_acceleration_cm_s2"]) for row in points],
            facecolors="none",
            edgecolors="black",
            s=35,
            label="selected data",
        )
        shape_ax.set_xscale("log")
        shape_ax.set_yscale("log")
        shape_ax.set_title(f"Condition {condition}: fixed bmax/aH=1")
        shape_ax.set_ylabel("acceleration (cm/s²)")
        shape_ax.legend(fontsize=8)
        shape_ax.grid(which="both", alpha=0.25)

        slope_ax = axes[1, column]
        slope_ax.plot(velocity, slope, color=color, linewidth=2)
        slope_ax.axhline(0.0, color="black", linewidth=0.8)
        slope_ax.set_xscale("log")
        slope_ax.set_ylabel("local slope d log(a) / d log(v)")
        slope_ax.grid(which="both", alpha=0.25)

        residual_ax = axes[2, column]
        residual_ax.scatter(
            [float(row["velocity_cm_s"]) for row in points],
            [float(row["model_acceleration_cm_s2"]) / float(row["data_acceleration_cm_s2"]) for row in points],
            color=color,
            s=35,
        )
        residual_ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
        residual_ax.set_xscale("log")
        residual_ax.set_yscale("log")
        residual_ax.set_xlabel("velocity (cm/s)")
        residual_ax.set_ylabel("model / data")
        residual_ax.grid(which="both", alpha=0.25)

    fig.suptitle("Fixed bmax/aH=1 drag-curve shape and residual diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare all four conditions with one fixed bmax/aH=1 model (no parameter optimization)."
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data-csv", type=Path)
    parser.add_argument(
        "--lammps-results",
        type=Path,
        default=REPO_ROOT / "theory" / "dataprocessing" / "output" / "results.npy",
    )
    parser.add_argument("--conditions", nargs="+", type=int, default=list(ALL_CONDITIONS))
    parser.add_argument("--samples-per-lammps-fit", type=int, default=10)
    parser.add_argument("--comparison-points-per-condition", type=int, default=8)
    parser.add_argument("--min-velocity-cm-s", type=float, default=1.0e2)
    parser.add_argument("--max-velocity-cm-s", type=float, default=1.0e8)
    parser.add_argument("--max-relative-sigma", type=float, default=10.0)
    parser.add_argument("--curve-points", type=int, default=24)
    parser.add_argument("--vres", type=int, default=50)
    parser.add_argument("--rhores", type=int, default=180)
    parser.add_argument("--ures", type=int, default=180)
    parser.add_argument("--dphires", type=int, default=180)
    args = parser.parse_args()

    requested_conditions = set(args.conditions)
    if args.data_csv:
        all_points = load_points_from_csv(args.data_csv, requested_conditions)
    else:
        all_points = load_lammps_expfit_points(args.lammps_results, requested_conditions, args.samples_per_lammps_fit)
    all_points = filter_points(
        all_points, args.min_velocity_cm_s, args.max_velocity_cm_s, args.max_relative_sigma
    )
    conditions = sorted({point.condition for point in all_points})
    omitted = sorted(requested_conditions.difference(conditions))
    for condition in omitted:
        print(f"[condition omitted] condition={condition}: no usable retained data")
    if not conditions:
        raise SystemExit("No requested condition has usable retained data.")
    selected_points, _ = select_fit_points(all_points, args.comparison_points_per_condition)
    if {point.condition for point in selected_points} != set(conditions):
        raise SystemExit("Could not select comparison data for every available condition.")

    point_tasks = [
        (point.condition, FIXED_BMAX_OVER_AH, FIT_PARAMETER, point, args.vres, args.rhores, args.ures, args.dphires)
        for point in selected_points
    ]
    curve_velocities = condition_curve_velocities(all_points, args.curve_points)
    curve_tasks = [
        (condition, 1.0, FIXED_BMAX_OVER_AH, FIT_PARAMETER, float(velocity), args.vres, args.rhores, args.ures, args.dphires, False)
        for condition in conditions
        for velocity in curve_velocities[condition]
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        prediction_rows = list(pool.map(point_diagnostic, point_tasks))
        curve_rows = list(pool.map(run_curve_case, curve_tasks))

    summaries = [
        summarize(condition, [row for row in prediction_rows if int(row["condition"]) == condition])
        for condition in conditions
    ]
    write_csv(OUTDIR / "fixed_bmax_ah1_lammps_predictions.csv", prediction_rows)
    write_csv(OUTDIR / "fixed_bmax_ah1_model_curves.csv", curve_rows)
    write_csv(OUTDIR / "fixed_bmax_ah1_summary.csv", summaries)
    plot_comparison(
        all_points,
        selected_points,
        curve_rows,
        summaries,
        conditions,
        OUTDIR / "fixed_bmax_ah1_comparison.png",
    )
    plot_shape_diagnostics(
        prediction_rows,
        curve_rows,
        conditions,
        OUTDIR / "fixed_bmax_ah1_shape_diagnostics.png",
    )

    for row in summaries:
        print(
            f"condition {row['condition']}: mean |fractional error|="
            f"{float(row['mean_absolute_fractional_residual']):.4g}, RMS="
            f"{float(row['rms_fractional_residual']):.4g}, max="
            f"{float(row['maximum_absolute_fractional_residual']):.4g}"
        )


if __name__ == "__main__":
    freeze_support()
    main()
