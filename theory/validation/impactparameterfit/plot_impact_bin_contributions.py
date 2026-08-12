# Run from repository root:
# python .\theory\validation\impactparameterfit\plot_impact_bin_contributions.py
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
from commonfinite import CM_PER_S_TO_M_PER_S, DEFAULT_CUTOFF_RADIUS_FACTOR, condition_label, make_drag
from resolution_scaling import scaled_resolution_for_bmax

CONDITIONS = (0, 1, 2, 3)
DEFAULT_BMAX_OVER_AH = (0.1, 1.0, 10.0)
DEFAULT_VELOCITIES_CM_S = (1.0e7, 3.0e7, 1.0e8)


def positive_float_list(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated numbers") from exc
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("provide positive finite values")
    return values


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def speed_grid_and_weights(drag, drift_velocity_m_s: float) -> tuple[np.ndarray, np.ndarray, float]:
    sigmav = math.sqrt(drag.kb * drag.T / drag.mu)
    width = drag.vrel_sigma_width * sigmav
    vmin = drift_velocity_m_s - width
    vmax = drift_velocity_m_s + width
    speed_min = 0.0 if vmin <= 0.0 <= vmax else min(abs(vmin), abs(vmax))
    speed_max = max(abs(vmin), abs(vmax))
    if drag.vres < 1 or speed_max <= speed_min:
        return np.array([], dtype=float), np.array([], dtype=float), math.nan

    ds = (speed_max - speed_min) / drag.vres
    speeds = speed_min + (np.arange(drag.vres, dtype=float) + 0.5) * ds
    norm = math.sqrt(drag.mu / (2.0 * math.pi * drag.kb * drag.T))

    positive = np.zeros_like(speeds)
    negative = np.zeros_like(speeds)
    positive_mask = (vmin <= speeds) & (speeds <= vmax)
    negative_mask = (vmin <= -speeds) & (-speeds <= vmax)
    positive[positive_mask] = norm * np.exp(-drag.mu * np.square(speeds[positive_mask] - drift_velocity_m_s) / (2.0 * drag.kb * drag.T))
    negative[negative_mask] = norm * np.exp(-drag.mu * np.square(-speeds[negative_mask] - drift_velocity_m_s) / (2.0 * drag.kb * drag.T))
    return speeds, positive - negative, ds


def bin_contribution_rows(
    condition: int,
    drift_velocity_cm_s: float,
    bmax_over_aH: float,
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
    drift_velocity_m_s = drift_velocity_cm_s * CM_PER_S_TO_M_PER_S
    speeds, weights, ds = speed_grid_and_weights(drag, drift_velocity_m_s)
    p, dp = drag._finite_launch_grid()
    if len(speeds) == 0 or len(p) == 0:
        return []

    bin_integral = np.zeros(len(p), dtype=float)
    bin_scattering_weight = np.zeros(len(p), dtype=float)
    valid_counts = np.zeros(len(p), dtype=int)

    for speed, weight in zip(speeds, weights):
        if weight == 0.0:
            continue
        energy = 0.5 * drag.mu * speed**2 + drag.E0Y
        half_theta = drag.finite_scattering_half_angle(p, float(speed), energy)
        valid = np.isfinite(half_theta)
        if not np.any(valid):
            continue

        scattering_factor = 2.0 * np.square(np.sin(half_theta[valid]))
        increment = p[valid] * dp[valid] * speed**2 * weight * scattering_factor
        bin_integral[valid] += increment
        bin_scattering_weight[valid] += abs(weight) * scattering_factor
        valid_counts[valid] += 1

    force_contribution = 2.0 * math.pi * drag.nh * drag.mu * bin_integral * ds
    total_drag_n = float(np.sum(force_contribution))
    aH_m = drag.launch_radius() / DEFAULT_CUTOFF_RADIUS_FACTOR

    rows = []
    for index, (p_center, p_width, force_n) in enumerate(zip(p, dp, force_contribution)):
        dp_over_aH = float(p_width / aH_m)
        rows.append(
            {
                "condition": condition,
                "condition_label": condition_label(condition),
                "drift_velocity_cm_s": drift_velocity_cm_s,
                "drift_velocity_m_s": drift_velocity_m_s,
                "bmax_over_aH": bmax_over_aH,
                "bin_index": index,
                "bin_count": len(p),
                "p_start_m": float(p_center),
                "p_start_over_aH": float(p_center / aH_m),
                "dp_start_m": float(p_width),
                "dp_start_over_aH": dp_over_aH,
                "bin_force_contribution_N": float(force_n),
                "absolute_bin_force_contribution_N": float(abs(force_n)),
                "absolute_force_density_per_aH_N": float(abs(force_n) / dp_over_aH) if dp_over_aH > 0.0 else math.nan,
                "total_drag_N_from_bins": total_drag_n,
                "fraction_of_total_drag": float(force_n / total_drag_n) if total_drag_n != 0.0 else math.nan,
                "signed_speed_integral_before_prefactor": float(bin_integral[index]),
                "mean_weighted_scattering_factor": float(bin_scattering_weight[index] / valid_counts[index])
                if valid_counts[index] > 0
                else math.nan,
                "valid_speed_count": int(valid_counts[index]),
                "hydrogen_interparticle_spacing_m": aH_m,
                "finite_launch_radius_m": drag.launch_radius(),
                "impact_parameter_cutoff_m": drag.launch_pmax(),
                "base_rhores_at_bmax_over_aH_1": resolution["rhores"],
                "base_dphires_at_bmax_over_aH_1": resolution["dphires"],
                **scaled_resolution,
            }
        )
    return rows


def build_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    resolution = {name: getattr(args, name) for name in ("vres", "rhores", "ures", "dphires")}
    rows = []
    for condition in args.conditions:
        for velocity_cm_s in args.velocities_cm_s:
            for bmax_over_aH in args.bmax_over_aH:
                rows.extend(
                    bin_contribution_rows(
                        condition=int(condition),
                        drift_velocity_cm_s=float(velocity_cm_s),
                        bmax_over_aH=float(bmax_over_aH),
                        resolution=resolution,
                    )
                )
    return rows


def plot_rows(rows: list[dict[str, object]], output_png: Path) -> None:
    velocities = sorted({float(row["drift_velocity_cm_s"]) for row in rows})
    bmax_values = sorted({float(row["bmax_over_aH"]) for row in rows})
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(bmax_values)))
    markers = ["o", "s", "^", "D", "P", "X"]
    fig, axes = plt.subplots(2, len(velocities), figsize=(5.4 * len(velocities), 8.2), sharex=True, squeeze=False)

    condition = int(rows[0]["condition"])
    for column, velocity_cm_s in enumerate(velocities):
        bin_axis = axes[0, column]
        density_axis = axes[1, column]
        for draw_order, (color, marker, bmax_over_aH) in enumerate(zip(colors, markers, bmax_values)):
            curve = sorted(
                (
                    row for row in rows
                    if float(row["drift_velocity_cm_s"]) == velocity_cm_s
                    and float(row["bmax_over_aH"]) == bmax_over_aH
                ),
                key=lambda row: float(row["p_start_over_aH"]),
            )
            x = np.array([float(row["p_start_over_aH"]) for row in curve], dtype=float)
            bin_force = np.array([float(row["absolute_bin_force_contribution_N"]) for row in curve], dtype=float)
            density = np.array([float(row["absolute_force_density_per_aH_N"]) for row in curve], dtype=float)
            total = float(curve[0]["total_drag_N_from_bins"]) if curve else math.nan
            bin_valid = np.isfinite(bin_force) & (bin_force > 0.0)
            density_valid = np.isfinite(density) & (density > 0.0)
            label = rf"$b_{{max}}/a_H={bmax_over_aH:g}$, total={abs(total):.2e} N"
            marker_size = 42 if len(curve) <= 20 else 14
            zorder = len(bmax_values) - draw_order
            jitter = 10.0 ** (0.018 * (draw_order - (len(bmax_values) - 1) / 2.0))
            x_plot = x * jitter
            bin_axis.scatter(
                x_plot[bin_valid],
                bin_force[bin_valid],
                facecolors="none",
                edgecolors=color,
                marker=marker,
                s=marker_size,
                alpha=0.95,
                linewidths=0.9,
                label=label,
                zorder=zorder,
            )
            density_axis.scatter(
                x_plot[density_valid],
                density[density_valid],
                facecolors="none",
                edgecolors=color,
                marker=marker,
                s=marker_size,
                alpha=0.95,
                linewidths=0.9,
                label=label,
                zorder=zorder,
            )
            for axis in (bin_axis, density_axis):
                axis.axvline(bmax_over_aH, color=color, linewidth=0.9, alpha=0.35)

        for axis in (bin_axis, density_axis):
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.25)
            axis.legend(fontsize=8)
        bin_axis.set_title(f"v={velocity_cm_s:.0e} cm/s")
        density_axis.set_xlabel("impact parameter at start p/aH (markers offset slightly to show overlap)")

    axes[0, 0].set_ylabel("|drag force from bin| [N]")
    axes[1, 0].set_ylabel("|dF / d(p/aH)| [N]")
    base_rhores = int(rows[0]["base_rhores_at_bmax_over_aH_1"])
    base_dphires = int(rows[0]["base_dphires_at_bmax_over_aH_1"])
    fig.suptitle(
        f"Condition {condition}: {condition_label(condition)}, drag contribution by launch-impact bin; "
        rf"$N_\rho={base_rhores:g}(b_{{max}}/a_H)^2$, "
        rf"$N_\phi={base_dphires:g}(b_{{max}}/a_H)$"
    )
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose finite-launch drag into contributions from launch impact-parameter bins."
    )
    parser.add_argument("--conditions", nargs="+", type=int, choices=CONDITIONS, default=[3])
    parser.add_argument("--velocities-cm-s", type=positive_float_list, default=DEFAULT_VELOCITIES_CM_S)
    parser.add_argument("--bmax-over-aH", type=positive_float_list, default=DEFAULT_BMAX_OVER_AH)
    parser.add_argument("--vres", type=int, default=50)
    parser.add_argument("--rhores", type=int, default=120, help="Impact-parameter bin count at bmax/aH=1.")
    parser.add_argument("--ures", type=int, default=120)
    parser.add_argument("--dphires", type=int, default=120)
    parser.add_argument("--output-csv", type=Path, default=OUTDIR / "impact_bin_contributions.csv")
    parser.add_argument("--output-png", type=Path, default=OUTDIR / "condition_3_impact_bin_contributions.png")
    args = parser.parse_args()

    if max(args.bmax_over_aH) > DEFAULT_CUTOFF_RADIUS_FACTOR:
        parser.error(f"--bmax-over-aH values cannot exceed {DEFAULT_CUTOFF_RADIUS_FACTOR:g}")
    if args.vres < 1 or args.rhores < 1 or args.ures < 2 or args.dphires < 2:
        parser.error("resolution values must be positive, with ures/dphires at least 2")

    rows = build_rows(args)
    if not rows:
        raise SystemExit("No bin contribution rows were generated.")
    write_rows(args.output_csv, rows)
    plot_rows(rows, args.output_png)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
