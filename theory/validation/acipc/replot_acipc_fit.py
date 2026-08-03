from __future__ import annotations

import csv
import math
import os
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUTDIR / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def is_ok(row: dict[str, object]) -> bool:
    return row.get("status") == "ok"


def bmax_over_interatomic_spacing(summary: dict[str, object]) -> float:
    bmax_m = as_float(summary.get("impact_parameter_cutoff_m"))
    outer_radius_m = as_float(summary.get("outer_radius_m"))
    cutoff_radius_factor = as_float(summary.get("cutoff_radius_factor"))
    if np.isfinite(bmax_m) and np.isfinite(outer_radius_m) and np.isfinite(cutoff_radius_factor) and cutoff_radius_factor > 0.0:
        return bmax_m / (outer_radius_m / cutoff_radius_factor)
    return as_float(summary.get("rhomax_fraction_of_interparticle_spacing"))


def main() -> None:
    summary_rows = read_rows(OUTDIR / "acipc_fit_summary.csv")
    prediction_rows = read_rows(OUTDIR / "acipc_fit_predictions.csv")
    curve_rows = read_rows(OUTDIR / "acipc_fit_curve.csv")

    colors = {0: "red", 1: "orange", 2: "green", 3: "blue"}
    fig, ax = plt.subplots(figsize=(12, 8))
    conditions = sorted({int(row["condition"]) for row in prediction_rows + curve_rows if row.get("condition", "") != ""})
    for condition in conditions:
        color = colors.get(condition)
        predictions = [row for row in prediction_rows if int(row["condition"]) == condition and is_ok(row)]
        curve = [row for row in curve_rows if int(row["condition"]) == condition and is_ok(row)]
        summary = next((row for row in summary_rows if int(row["condition"]) == condition and is_ok(row)), None)
        curve.sort(key=lambda row: as_float(row["velocity_cm_s"]))

        ax.errorbar(
            [as_float(row["velocity_cm_s"]) for row in predictions],
            [as_float(row["data_acceleration_cm_s2"]) for row in predictions],
            yerr=[
                as_float(row["data_acceleration_sigma_cm_s2"])
                if np.isfinite(as_float(row["data_acceleration_sigma_cm_s2"]))
                else 0.0
                for row in predictions
            ],
            fmt="o",
            color=color,
            mfc="none",
            mec=color,
            markersize=7,
            markeredgewidth=1.5,
            alpha=0.85,
            linewidth=1.0,
            capsize=0,
        )

        if curve and summary:
            label = (
                f"condition {condition} "
                f"acipc={as_float(summary['acipc']):.3g}, "
                f"bmax/aH={bmax_over_interatomic_spacing(summary):.3g}"
            )
            ax.plot(
                [as_float(row["velocity_cm_s"]) for row in curve],
                [as_float(row["model_acceleration_cm_s2"]) for row in curve],
                color=color,
                linewidth=2.0,
                label=label,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("velocity [cm/s]")
    ax.set_ylabel("acceleration [cm/s^2]")
    ax.set_title("Two-parameter fit: acipc and impact-parameter cutoff")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "acipc_fit.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
