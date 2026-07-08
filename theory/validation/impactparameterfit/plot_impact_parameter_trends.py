from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUTDIR / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


NORMALIZATIONS = [
    (
        "bmax/aH",
        "rhomax_fraction_of_interparticle_spacing",
        "rhomax_fraction_of_interparticle_spacing_sigma",
    ),
    (
        "bmax/lion",
        "rhomax_fraction_of_ion_screening_length",
        "rhomax_fraction_of_ion_screening_length_sigma",
    ),
    (
        "bmax/lY",
        "rhomax_fraction_of_yukawa_screening_length",
        "rhomax_fraction_of_yukawa_screening_length_sigma",
    ),
    (
        "bmax/lD(e)",
        "rhomax_fraction_of_debye_length",
        "rhomax_fraction_of_debye_length_sigma",
    ),
]


def finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if np.isfinite(result) else math.nan


def load_summary(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"No rows found in {path}")
    return sorted(rows, key=lambda row: finite_float(row.get("coupling_parameter")))


def write_trend_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "condition",
        "condition_label",
        "coupling_parameter",
        "fit_point_selection",
        "bmax_m",
        "bmax/aH",
        "bmax/aH_sigma",
        "bmax/lion",
        "bmax/lion_sigma",
        "bmax/lY",
        "bmax/lY_sigma",
        "bmax/lD(e)",
        "bmax/lD(e)_sigma",
    ]
    output_rows = []
    for row in rows:
        output_rows.append(
            {
                "condition": row.get("condition", ""),
                "condition_label": row.get("condition_label", ""),
                "coupling_parameter": row.get("coupling_parameter", ""),
                "fit_point_selection": row.get("fit_point_selection", ""),
                "bmax_m": row.get("impact_parameter_upper_bound_m", ""),
                "bmax/aH": row.get("rhomax_fraction_of_interparticle_spacing", ""),
                "bmax/aH_sigma": row.get("rhomax_fraction_of_interparticle_spacing_sigma", ""),
                "bmax/lion": row.get("rhomax_fraction_of_ion_screening_length", ""),
                "bmax/lion_sigma": row.get("rhomax_fraction_of_ion_screening_length_sigma", ""),
                "bmax/lY": row.get("rhomax_fraction_of_yukawa_screening_length", ""),
                "bmax/lY_sigma": row.get("rhomax_fraction_of_yukawa_screening_length_sigma", ""),
                "bmax/lD(e)": row.get("rhomax_fraction_of_debye_length", ""),
                "bmax/lD(e)_sigma": row.get("rhomax_fraction_of_debye_length_sigma", ""),
            }
        )
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def plot_trends(rows: list[dict[str, object]], output_png: Path) -> None:
    gamma = np.array([finite_float(row.get("coupling_parameter")) for row in rows], dtype=float)
    conditions = [str(row.get("condition", "")) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, value_key, sigma_key in NORMALIZATIONS:
        values = np.array([finite_float(row.get(value_key)) for row in rows], dtype=float)
        sigmas = np.array([finite_float(row.get(sigma_key)) for row in rows], dtype=float)
        mask = np.isfinite(gamma) & np.isfinite(values) & (gamma > 0.0) & (values > 0.0)
        if not np.any(mask):
            continue
        yerr = np.where(np.isfinite(sigmas[mask]), sigmas[mask], 0.0)
        ax.errorbar(
            gamma[mask],
            values[mask],
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=1.8,
            label=label,
        )

    for row_gamma, condition in zip(gamma, conditions):
        if np.isfinite(row_gamma):
            ax.annotate(f"i={condition}", (row_gamma, 0.95), xycoords=("data", "axes fraction"), ha="center", fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Plasma coupling parameter Gamma")
    ax.set_ylabel("Best-fit impact parameter normalization")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, default=OUTDIR / "impact_parameter_fit_summary.csv")
    parser.add_argument("--output-png", type=Path, default=OUTDIR / "impact_parameter_fit_trends.png")
    parser.add_argument("--output-csv", type=Path, default=OUTDIR / "impact_parameter_fit_trends.csv")
    args = parser.parse_args()

    rows = load_summary(args.summary_csv)
    write_trend_csv(args.output_csv, rows)
    plot_trends(rows, args.output_png)
    print(f"Wrote {args.output_png}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
