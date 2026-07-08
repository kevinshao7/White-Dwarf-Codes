from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CM_PER_S_TO_M_PER_S, add_common_args, condition_label, make_drag, quiet_drag, velocity_cases, write_csv

OUTDIR = Path(__file__).resolve().parent
VELOCITY_CM_S = np.array([
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    1.0e3,
    1.0e4,
    1.0e5,
    1.0e6,
    3.0e6,
    5.0e6,
    1.0e7,
    2.0e7,
    5.0e7,
    1.0e8,
])


def run_case(task: tuple[int, float, int, int, int, int]) -> dict[str, float | int]:
    condition, velocity_cm_s, vres, rhores, ures, dphires = task
    drag = make_drag(condition, vres, rhores, ures, dphires)
    velocity_m_s = velocity_cm_s * CM_PER_S_TO_M_PER_S
    return {
        "condition": condition,
        "velocity_cm_s": velocity_cm_s,
        "velocity_m_s": velocity_m_s,
        "drag_N": quiet_drag(drag, velocity_m_s),
    }


def log_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(x), np.log(np.abs(y)), 1)[0])


def high_velocity_tail(velocities: np.ndarray, drags: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    peak_idx = int(np.argmax(np.abs(drags)))
    tail_velocities = []
    tail_drags = []
    previous_abs_drag = abs(drags[peak_idx])
    for velocity, drag in zip(velocities[peak_idx + 1 :], drags[peak_idx + 1 :]):
        abs_drag = abs(drag)
        if abs_drag > previous_abs_drag and len(tail_velocities) >= 3:
            break
        tail_velocities.append(velocity)
        tail_drags.append(drag)
        previous_abs_drag = abs_drag
    tail_velocities = np.array(tail_velocities, dtype=float)
    tail_drags = np.array(tail_drags, dtype=float)
    if len(tail_velocities) < 3:
        tail_velocities = velocities[peak_idx + 1 :]
        tail_drags = drags[peak_idx + 1 :]
    if len(tail_velocities) < 3:
        tail_velocities = velocities[-3:]
        tail_drags = drags[-3:]
    return tail_velocities, tail_drags, float(velocities[peak_idx])


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()

    tasks = [
        (condition, float(velocity), args.vres, args.rhores, args.ures, args.dphires)
        for condition in args.conditions
        for velocity in VELOCITY_CM_S
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_case, tasks))

    rows.sort(key=lambda r: (r["condition"], r["velocity_cm_s"]))
    for condition in args.conditions:
        group = [row for row in rows if row["condition"] == condition]
        velocities = np.array([row["velocity_cm_s"] for row in group], dtype=float)
        drags = np.array([row["drag_N"] for row in group], dtype=float)
        low_slope = log_slope(velocities[:4], drags[:4])
        high_velocities, high_drags, bragg_peak_cm_s = high_velocity_tail(velocities, drags)
        high_slope = log_slope(high_velocities, high_drags)
        thermal_cm_s = velocity_cases(make_drag(condition))[("thermal_1d")] / CM_PER_S_TO_M_PER_S
        for row in group:
            row["thermal_velocity_cm_s"] = thermal_cm_s
            row["bragg_peak_velocity_cm_s"] = bragg_peak_cm_s
            row["high_fit_min_velocity_cm_s"] = float(high_velocities[0])
            row["high_fit_max_velocity_cm_s"] = float(high_velocities[-1])
            row["high_fit_region"] = "contiguous decreasing branch above Bragg peak"
            row["used_in_high_velocity_fit"] = row["velocity_cm_s"] in set(high_velocities)
            row["low_velocity_slope_first_4"] = low_slope
            row["high_velocity_slope_above_bragg_peak"] = high_slope
            row["low_expected_slope"] = 1.0
            row["high_expected_slope"] = -2.0
            row["drag_over_v"] = row["drag_N"] / row["velocity_m_s"]
            row["drag_times_v2"] = row["drag_N"] * row["velocity_m_s"] ** 2

    write_csv(OUTDIR / "velocity_limits.csv", rows)

    fig, axes = plt.subplots(len(args.conditions), 1, figsize=(7, 3 * len(args.conditions)), squeeze=False)
    for ax, condition in zip(axes[:, 0], args.conditions):
        group = [row for row in rows if row["condition"] == condition]
        ax.plot(
            [row["velocity_cm_s"] for row in group],
            [abs(row["drag_N"]) for row in group],
            marker="o",
        )
        thermal = group[0]["thermal_velocity_cm_s"]
        bragg_peak = group[0]["bragg_peak_velocity_cm_s"]
        high_fit_min = group[0]["high_fit_min_velocity_cm_s"]
        ax.axvline(thermal, color="k", linestyle="--", linewidth=1, label="thermal velocity")
        ax.axvline(bragg_peak, color="0.4", linestyle=":", linewidth=1, label="Bragg peak")
        ax.axvspan(high_fit_min, max(row["velocity_cm_s"] for row in group), color="0.9", alpha=0.5, label="high-v fit")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("velocity [cm/s]")
        ax.set_ylabel("|drag| [N]")
        ax.set_title(
            f"{condition_label(condition)}: low slope {group[0]['low_velocity_slope_first_4']:.2f}, "
            f"high slope {group[0]['high_velocity_slope_above_bragg_peak']:.2f}"
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "velocity_limits.png", dpi=200)


if __name__ == "__main__":
    freeze_support()
    main()
