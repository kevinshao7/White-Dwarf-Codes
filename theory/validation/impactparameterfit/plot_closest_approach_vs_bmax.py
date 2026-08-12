# Run from repository root:
# python .\theory\validation\impactparameterfit\plot_closest_approach_vs_bmax.py
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUTDIR / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CM_PER_S_TO_M_PER_S, DEFAULT_CUTOFF_RADIUS_FACTOR, condition_label, make_drag
from resolution_scaling import scaled_resolution_for_bmax

CONDITIONS = (0, 1, 2, 3)
DEFAULT_VELOCITIES_CM_S = (1.0e5, 1.0e6, 1.0e7)
DEFAULT_BIN_FRACTIONS = (0.1, 0.5, 0.9)


def positive_float_csv(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated numbers") from exc
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive finite numbers")
    return values


def fraction_csv(text: str) -> tuple[float, ...]:
    values = positive_float_csv(text)
    if any(value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("bin fractions must satisfy 0 < value <= 1")
    return values


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def equal_area_rhoinf_grid(rhoup: float, count: int) -> np.ndarray:
    rhoarr = np.zeros(count, dtype=float)
    for index in range(count):
        if index == 0:
            rhoarr[0] = 0.5 * math.sqrt(rhoup**2 / count)
        else:
            rhoarr[index] = math.sqrt(rhoup**2 / count + rhoarr[index - 1] ** 2)
    return rhoarr


def retained_bin_indices(retained_count: int, bin_fractions: tuple[float, ...]) -> list[tuple[float, int]]:
    if retained_count < 1:
        return []
    indices = []
    for fraction in bin_fractions:
        index = int(round(fraction * (retained_count - 1)))
        index = min(max(index, 0), retained_count - 1)
        indices.append((fraction, index))
    return indices


def closest_approach_rows(
    condition: int,
    velocity_cm_s: float,
    bmax_over_aH: float,
    bin_fractions: tuple[float, ...],
    resolution: dict[str, int],
) -> list[dict[str, object]]:
    rhomax_fraction = bmax_over_aH / DEFAULT_CUTOFF_RADIUS_FACTOR
    scaled_resolution = scaled_resolution_for_bmax(resolution, bmax_over_aH)
    drag = make_drag(
        condition,
        rhomax_fraction=rhomax_fraction,
        cutoff_radius_factor=DEFAULT_CUTOFF_RADIUS_FACTOR,
        **scaled_resolution,
    )

    speed = velocity_cm_s * CM_PER_S_TO_M_PER_S
    energy = 0.5 * drag.mu * speed**2 + drag.E0Y
    vinf = math.sqrt(energy / (0.5 * drag.mu))
    rhoup = rhomax_fraction * speed / (vinf * drag.ustart)
    rhoinf_grid = equal_area_rhoinf_grid(rhoup, drag.rhores)
    vstartphi = rhoinf_grid * vinf * drag.ustart
    alpha = np.arcsin(vstartphi / speed)
    rhostart_grid = np.sin(alpha) / drag.ustart
    bmax_m = rhomax_fraction / drag.ustart
    keep = rhostart_grid <= bmax_m
    if not np.any(keep):
        return []
    retained_count = int(np.where(keep)[0][-1]) + 1
    selected = retained_bin_indices(retained_count, bin_fractions)
    selected_rhoinf = np.array([rhoinf_grid[index] for _, index in selected], dtype=float)
    u0_values = drag.umax(selected_rhoinf, energy)

    outer_radius_m = 1.0 / drag.ustart
    hydrogen_spacing_m = outer_radius_m / DEFAULT_CUTOFF_RADIUS_FACTOR
    rows = []
    for (bin_fraction, index), rhoinf, u0 in zip(selected, selected_rhoinf, u0_values):
        closest_approach_m = 1.0 / float(u0)
        rhostart_m = float(rhostart_grid[index])
        rows.append(
            {
                "condition": condition,
                "condition_label": condition_label(condition),
                "velocity_cm_s": velocity_cm_s,
                "velocity_m_s": speed,
                "bmax_over_aH": bmax_over_aH,
                "retained_bin_fraction": bin_fraction,
                "retained_bin_index": index,
                "retained_bin_count": retained_count,
                "rhomax_fraction_of_launch_radius": rhomax_fraction,
                "launch_radius_over_aH": DEFAULT_CUTOFF_RADIUS_FACTOR,
                "outer_radius_m": outer_radius_m,
                "hydrogen_interparticle_spacing_m": hydrogen_spacing_m,
                "bmax_m": bmax_m,
                "angle_cutoff_radius_m": bmax_m,
                "rhoinf_m": float(rhoinf),
                "rhostart_m": rhostart_m,
                "rhostart_over_aH": rhostart_m / hydrogen_spacing_m,
                "rhostart_over_bmax": rhostart_m / bmax_m,
                "closest_approach_m": closest_approach_m,
                "closest_approach_over_aH": closest_approach_m / hydrogen_spacing_m,
                "closest_approach_over_bmax": closest_approach_m / bmax_m,
                "u0_inverse_closest_approach_m^-1": float(u0),
                "energy_J": energy,
                "base_rhores_at_bmax_over_aH_1": resolution["rhores"],
                "base_dphires_at_bmax_over_aH_1": resolution["dphires"],
                **scaled_resolution,
            }
        )
    return rows


def build_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    bmax_values = np.geomspace(args.min_bmax_over_aH, args.max_bmax_over_aH, args.bmax_points)
    resolution = {name: getattr(args, name) for name in ("vres", "rhores", "ures", "dphires")}
    rows = []
    for condition in args.conditions:
        for velocity_cm_s in args.velocities_cm_s:
            for bmax_over_aH in bmax_values:
                rows.extend(
                    closest_approach_rows(
                        condition=int(condition),
                        velocity_cm_s=float(velocity_cm_s),
                        bmax_over_aH=float(bmax_over_aH),
                        bin_fractions=args.bin_fractions,
                        resolution=resolution,
                    ),
                )
    return rows


def plot_rows(rows: list[dict[str, object]], output_png: Path) -> None:
    conditions = sorted({int(row["condition"]) for row in rows})
    velocities = sorted({float(row["velocity_cm_s"]) for row in rows})
    bin_fractions = sorted({float(row["retained_bin_fraction"]) for row in rows})
    colors = plt.cm.plasma(np.linspace(0.12, 0.86, len(velocities)))
    linestyles = ["-", "--", ":", "-."]
    fig, axes = plt.subplots(1, len(conditions), figsize=(5.2 * len(conditions), 4.8), squeeze=False)

    for axis, condition in zip(axes.flat, conditions):
        condition_rows = [row for row in rows if int(row["condition"]) == condition]
        for color, velocity_cm_s in zip(colors, velocities):
            for linestyle, bin_fraction in zip(linestyles, bin_fractions):
                curve = sorted(
                    (
                        row for row in condition_rows
                        if float(row["velocity_cm_s"]) == velocity_cm_s
                        and float(row["retained_bin_fraction"]) == bin_fraction
                    ),
                    key=lambda row: float(row["bmax_over_aH"]),
                )
                x = np.array([float(row["bmax_over_aH"]) for row in curve], dtype=float)
                y = np.array([float(row["closest_approach_over_aH"]) for row in curve], dtype=float)
                axis.plot(
                    x,
                    y,
                    marker="o",
                    markersize=2.6,
                    linewidth=1.5,
                    color=color,
                    linestyle=linestyle,
                    label=f"{velocity_cm_s:.0e} cm/s, bin {bin_fraction:.1g}",
                )

        reference = sorted(condition_rows, key=lambda row: float(row["bmax_over_aH"]))
        x_reference = np.unique(np.array([float(row["bmax_over_aH"]) for row in reference], dtype=float))
        axis.plot(x_reference, x_reference, color="black", linestyle="--", linewidth=1.0, label="rmin = bmax")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("impact-parameter cutoff bmax/aH")
        axis.set_ylabel("closest approach rmin/aH")
        axis.set_title(f"Condition {condition}: {condition_label(condition)}")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)

    fig.suptitle("Closest approach for retained impact-parameter bins")
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot rmin versus bmax/aH for selected retained impact-parameter bins. "
            "The finite scattering-angle radial cutoff equals bmax because acipc=1."
        )
    )
    parser.add_argument("--conditions", nargs="+", type=int, choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--velocities-cm-s", type=positive_float_csv, default=DEFAULT_VELOCITIES_CM_S)
    parser.add_argument(
        "--bin-fractions",
        type=fraction_csv,
        default=DEFAULT_BIN_FRACTIONS,
        help="Comma-separated retained-grid positions to sample, with 1 being the outermost retained bin.",
    )
    parser.add_argument("--min-bmax-over-aH", type=float, default=0.05)
    parser.add_argument("--max-bmax-over-aH", type=float, default=DEFAULT_CUTOFF_RADIUS_FACTOR)
    parser.add_argument("--bmax-points", type=int, default=40)
    parser.add_argument("--vres", type=int, default=50)
    parser.add_argument("--rhores", type=int, default=180, help="Impact-parameter bin count at bmax/aH=1.")
    parser.add_argument("--ures", type=int, default=180)
    parser.add_argument("--dphires", type=int, default=180)
    parser.add_argument("--output-csv", type=Path, default=OUTDIR / "closest_approach_vs_bmax.csv")
    parser.add_argument("--output-png", type=Path, default=OUTDIR / "closest_approach_vs_bmax.png")
    args = parser.parse_args()

    if args.min_bmax_over_aH <= 0.0:
        parser.error("--min-bmax-over-aH must be positive")
    if args.max_bmax_over_aH <= args.min_bmax_over_aH:
        parser.error("--max-bmax-over-aH must be greater than --min-bmax-over-aH")
    if args.max_bmax_over_aH > DEFAULT_CUTOFF_RADIUS_FACTOR:
        parser.error(f"--max-bmax-over-aH cannot exceed {DEFAULT_CUTOFF_RADIUS_FACTOR:g}, the launch radius in aH")
    if args.bmax_points < 2:
        parser.error("--bmax-points must be at least 2")
    if args.rhores < 1:
        parser.error("--rhores is the bin count at bmax/aH=1 and must be positive")

    rows = build_rows(args)
    write_rows(args.output_csv, rows)
    plot_rows(rows, args.output_png)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
