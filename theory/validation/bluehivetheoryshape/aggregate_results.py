from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hpc_shape_common import BMAX_OVER_AH, OUTDIR, RESULTS_DIR, add_shape_columns, condition_label, read_rows_csv, write_rows_csv


def finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def load_result_rows(results_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(results_dir.glob("task_*.csv")):
        rows.extend(read_rows_csv(path))
    if not rows:
        raise SystemExit(f"No task_*.csv files found in {results_dir}")
    return rows


def plot_condition(rows: list[dict[str, object]], condition: int, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(BMAX_OVER_AH)))

    for color, bmax in zip(colors, BMAX_OVER_AH):
        curve = sorted(
            (
                row
                for row in rows
                if int(row["condition"]) == condition
                and float(row["bmax_over_hydrogen_interparticle_spacing"]) == bmax
            ),
            key=lambda row: float(row["velocity_cm_s"]),
        )
        velocity = np.array([finite_float(row["velocity_cm_s"]) for row in curve], dtype=float)
        drag = np.array([finite_float(row["absolute_drag_N"]) for row in curve], dtype=float)
        normalized = np.array([finite_float(row["drag_normalized_to_peak"]) for row in curve], dtype=float)
        valid = np.isfinite(velocity) & np.isfinite(drag) & (drag > 0.0)
        normalized_valid = valid & np.isfinite(normalized) & (normalized > 0.0)
        label = rf"$b_{{max}}/a_H={bmax:g}$"

        axes[0].plot(velocity[valid], drag[valid], marker="o", linewidth=1.8, markersize=3.5, color=color, label=label)
        axes[1].plot(
            velocity[normalized_valid],
            normalized[normalized_valid],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
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
        axis.legend(fontsize=7)

    fig.suptitle(f"Condition {condition}: {condition_label(condition)}, BlueHive shape sweep")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate BlueHive impact-parameter shape task outputs.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-csv", type=Path, default=OUTDIR / "bluehive_impact_parameter_shape_curves.csv")
    args = parser.parse_args()

    rows = load_result_rows(args.results_dir)
    rows.sort(
        key=lambda row: (
            int(row["condition"]),
            float(row["bmax_over_hydrogen_interparticle_spacing"]),
            float(row["velocity_cm_s"]),
        )
    )
    add_shape_columns(rows)
    write_rows_csv(args.output_csv, rows)
    print(f"Wrote {args.output_csv}")

    for condition in sorted({int(row["condition"]) for row in rows}):
        output_png = OUTDIR / f"condition_{condition}_bluehive_impact_parameter_shape.png"
        plot_condition(rows, condition, output_png)
        print(f"Wrote {output_png}")


if __name__ == "__main__":
    main()
