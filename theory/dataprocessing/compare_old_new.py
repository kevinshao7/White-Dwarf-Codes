# Run from repository root: python .\theory\dataprocessing\compare_old_new.py
"""Compare legacy and reliability-filtered unforced velocity-decay fits."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def valid_rows(array: np.ndarray) -> np.ndarray:
    return np.isfinite(array).all(axis=2) & (array > 0.0).all(axis=2)


def decay_curve(row: np.ndarray, samples: int = 40) -> tuple[np.ndarray, np.ndarray]:
    amplitude, _, tau, _, start_time, end_time = row
    times = np.linspace(start_time, end_time, samples)
    velocity = amplitude * np.exp(-times / tau)
    acceleration = velocity / tau
    mask = np.isfinite(velocity) & np.isfinite(acceleration) & (velocity > 0.0) & (acceleration > 0.0)
    return velocity[mask], acceleration[mask]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old",
        type=Path,
        default=repo_root / "unforced" / "dataarchive" / "nprun4_29" / "results.npy",
    )
    parser.add_argument(
        "--new",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "results.npy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "old_vs_new_processing.png",
    )
    args = parser.parse_args()

    old = np.load(args.old, allow_pickle=False)
    new = np.load(args.new, allow_pickle=False)
    if old.shape != new.shape or old.ndim != 3 or old.shape[2] < 6:
        raise SystemExit(f"Incompatible arrays: old={old.shape}, new={new.shape}; expected matching (condition, campaign, 6)")

    old_valid = valid_rows(old)
    new_valid = valid_rows(new)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#d32f2f", "#f57c00", "#388e3c", "#1976d2"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    for condition in range(4):
        curve_ax = axes[condition]

        first_old = True
        for row in old[condition, old_valid[condition]]:
            velocity, acceleration = decay_curve(row)
            if velocity.size:
                curve_ax.plot(
                    velocity,
                    acceleration,
                    color="0.65",
                    alpha=0.45,
                    linewidth=0.8,
                    label="legacy retained fits" if first_old else None,
                )
                first_old = False

        first_new = True
        for row in new[condition, new_valid[condition]]:
            velocity, acceleration = decay_curve(row)
            if velocity.size:
                curve_ax.plot(
                    velocity,
                    acceleration,
                    color=colors[condition],
                    alpha=0.9,
                    linewidth=1.5,
                    label="new retained fits" if first_new else None,
                )
                first_new = False

        old_count = int(old_valid[condition].sum())
        new_count = int(new_valid[condition].sum())
        curve_ax.set_title(f"Condition {condition}: old {old_count}, new {new_count}", color=colors[condition])
        curve_ax.set_xscale("log")
        curve_ax.set_yscale("log")
        curve_ax.set_xlabel("velocity (cm/s)")
        if condition == 0:
            curve_ax.set_ylabel("acceleration (cm/s²)")
        curve_ax.grid(which="both", alpha=0.25)
        if old_count or new_count:
            curve_ax.legend(fontsize=8)
        if new_count == 0:
            curve_ax.text(
                0.5,
                0.5,
                "No newly retained fits",
                transform=curve_ax.transAxes,
                ha="center",
                va="center",
                color=colors[condition],
                fontsize=12,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": colors[condition]},
            )

    fig.suptitle("Acceleration versus velocity: legacy and newly analyzed data", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
