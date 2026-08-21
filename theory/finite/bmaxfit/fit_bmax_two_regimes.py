"""Fit two b_max/a_H values per condition: one below, one above thermal velocity.

`fit_bmax_to_lammps.py` (`theory/finite/lammps_fit/`) fits one `b_max/a_H`
per condition, held constant across the whole velocity range. This script
asks the coarsest possible velocity-dependent question: does one value fit
sub-thermal drift and a different one fit super-thermal drift? Selected
LAMMPS(+DAIS) points are split at each condition's thermal width
`v_th = sqrt(kB T / mu)` (the quantity `FiniteLaunchDrag.drag` calls
`sigma_v`) into a below-`v_th` and an above-`v_th` group, and
`fit_bmax_to_lammps.fit_condition` -- the same shared `least_squares` +
parallel-drag-evaluation machinery used for the single-value fit, completely
unmodified -- is run once per group. That gives 2 x 4 = 8 fitted values
instead of `fit_bmax_to_lammps`'s 4, directly comparable to it since each
regime fit uses the identical residual definition and bounds.

Parallelism is inherited from `fit_condition`: each regime fit's
`least_squares` iterations submit one drag evaluation per fit point to a
shared `ProcessPoolExecutor`, same as the single-value fit.

Run from repository root:
    python .\\theory\\finite\\bmaxfit\\fit_bmax_two_regimes.py --workers 8
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUTDIR / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

if str(OUTDIR) not in sys.path:
    sys.path.insert(0, str(OUTDIR))

import common  # noqa: E402
from common import base  # noqa: E402

REGIMES = ("below_thermal", "above_thermal")
REGIME_COLORS = {"below_thermal": "tab:blue", "above_thermal": "tab:red"}
DEFAULT_POINTS_PER_REGIME = 8
DEFAULT_MAX_NFEV = 30


def split_by_thermal_velocity(
    points: list[common.DataPoint], v_th_cm_s: float
) -> dict[str, list[common.DataPoint]]:
    below = [p for p in points if p.velocity_cm_s < v_th_cm_s]
    above = [p for p in points if p.velocity_cm_s >= v_th_cm_s]
    return {"below_thermal": below, "above_thermal": above}


def make_regime_plot(
    condition: int,
    v_th_cm_s: float,
    regime_summaries: dict[str, dict[str, object]],
    regime_prediction_rows: dict[str, list[dict[str, object]]],
    all_points: list[common.DataPoint],
    regime_uncertainty: dict[str, dict[str, list[dict[str, object]]]],
) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))

    cond_all_v = np.array([p.velocity_cm_s for p in all_points if p.condition == condition])
    cond_all_a = np.array([p.acceleration_cm_s2 for p in all_points if p.condition == condition])
    axis.scatter(cond_all_v, cond_all_a, s=8, color="lightgray", label="LAMMPS points (not fit)", zorder=1)

    for regime in REGIMES:
        summary = regime_summaries.get(regime)
        rows = regime_prediction_rows.get(regime, [])
        if summary is None or not rows:
            continue
        color = REGIME_COLORS[regime]
        fit_v = np.array([row["velocity_cm_s"] for row in rows])
        fit_a = np.array([row["data_acceleration_cm_s2"] for row in rows])
        fit_sigma = np.array([row["data_acceleration_sigma_cm_s2"] for row in rows])
        axis.errorbar(
            fit_v, fit_a, yerr=fit_sigma, fmt="o", color=color, markersize=6, capsize=3,
            label=f"{regime.replace('_', ' ')} fit points", zorder=4,
        )

        order = np.argsort(fit_v)
        model_v = fit_v[order]
        model_a = np.array([row["model_acceleration_cm_s2"] for row in rows])[order]

        unc = regime_uncertainty.get(regime) or {}
        low_rows, high_rows = unc.get("low"), unc.get("high")
        if low_rows and high_rows:
            low_a = np.array([row["model_acceleration_cm_s2"] for row in low_rows])[order]
            high_a = np.array([row["model_acceleration_cm_s2"] for row in high_rows])[order]
            axis.fill_between(
                model_v, np.minimum(low_a, high_a), np.maximum(low_a, high_a),
                color=color, alpha=0.15, linewidth=0, zorder=1.5,
            )

        label = f"{regime.replace('_', ' ')} model, $b_{{max}}/a_H={summary['best_bmax_over_aH']:.4g}$"
        axis.plot(model_v, model_a, color=color, linewidth=2.0, marker="s", markersize=4, label=label, zorder=3)

    axis.axvline(
        v_th_cm_s, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
        label=f"$v_{{th}}$ = {v_th_cm_s:.3g} cm/s",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("velocity [cm/s]")
    axis.set_ylabel("acceleration [cm/s^2]")
    axis.set_title(f"Condition {condition}: b_max fit split at thermal velocity")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"condition_{condition}_bmax_two_regimes_overlay.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = common.build_common_parser(__doc__)
    parser.add_argument(
        "--points-per-regime", type=int, default=DEFAULT_POINTS_PER_REGIME,
        help="quantile groups per regime per condition, same selection rule as fit_bmax_to_lammps.select_fit_points",
    )
    parser.add_argument("--max-nfev", type=int, default=DEFAULT_MAX_NFEV)
    args = parser.parse_args()
    gpu_devices = base.parse_gpu_devices(args.gpu_devices)

    all_points, filtered, conditions = common.load_and_filter_points(args)

    start = time.perf_counter()
    summaries: list[dict[str, object]] = []
    all_prediction_rows: list[dict[str, object]] = []
    # GPU dispatch never touches the CPU pool (see fit_condition/evaluate_points
    # in fit_bmax_to_lammps.py), so skip spinning up worker processes for nothing.
    with contextlib.nullcontext(None) if gpu_devices else ProcessPoolExecutor(max_workers=args.workers) as pool:
        for condition in sorted(conditions):
            v_th_cm_s = common.thermal_velocity_cm_s(condition)
            cond_filtered = [p for p in filtered if p.condition == condition]
            regime_points_raw = split_by_thermal_velocity(cond_filtered, v_th_cm_s)

            regime_summaries: dict[str, dict[str, object]] = {}
            regime_prediction_rows: dict[str, list[dict[str, object]]] = {}
            regime_uncertainty: dict[str, dict[str, list[dict[str, object]]]] = {}

            for regime, regime_points in regime_points_raw.items():
                fit_points = base.select_fit_points(regime_points, args.points_per_regime)
                if not fit_points:
                    print(
                        f"Condition {condition} [{regime}]: no points after selection "
                        f"(v_th={v_th_cm_s:.4g} cm/s), skipping.", flush=True,
                    )
                    continue
                print(
                    f"Condition {condition} [{regime}]: fitting b_max/a_H against "
                    f"{len(fit_points)} points (v_th={v_th_cm_s:.4g} cm/s).", flush=True,
                )
                summary, prediction_rows, uncertainty_rows = base.fit_condition(
                    pool, condition, fit_points, args.bmax_min, args.bmax_max, args.method,
                    args.resolution, args.vres, args.max_nfev,
                    progress=not args.quiet, heartbeat_seconds=args.heartbeat_seconds,
                    gpu_devices=gpu_devices,
                )
                summary["regime"] = regime
                summary["thermal_velocity_cm_s"] = v_th_cm_s

                if summary["at_upper_bound"]:
                    print(
                        f"  WARNING: condition {condition} [{regime}] best fit sits at the "
                        f"b_max/a_H upper bound ({args.bmax_max:g}).", flush=True,
                    )
                print(
                    f"  best b_max/a_H = {summary['best_bmax_over_aH']:.6g} "
                    f"+/- {summary['best_bmax_over_aH_sigma']:.3g}, "
                    f"reduced chi2 = {summary['reduced_chi2']:.4g}, converged={summary['converged']}.",
                    flush=True,
                )

                for row in prediction_rows:
                    row["regime"] = regime

                regime_summaries[regime] = summary
                regime_prediction_rows[regime] = prediction_rows
                regime_uncertainty[regime] = uncertainty_rows
                summaries.append(summary)
                all_prediction_rows.extend(prediction_rows)

            if regime_summaries:
                make_regime_plot(
                    condition, v_th_cm_s, regime_summaries, regime_prediction_rows,
                    all_points, regime_uncertainty,
                )

    base.write_csv(OUTDIR / "bmax_two_regimes_fit_summary.csv", summaries)
    base.write_csv(OUTDIR / "bmax_two_regimes_fit_predictions.csv", all_prediction_rows)
    print(f"Finished in {(time.perf_counter()-start)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    freeze_support()
    main()
