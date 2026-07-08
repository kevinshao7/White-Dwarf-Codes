from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_common_args, condition_label, make_drag, velocity_cases, write_csv

OUTDIR = Path(__file__).resolve().parent
K0_B90_VALUES = [1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4]
RHO_OVER_B90 = [0.25, 1.0, 4.0]


def rutherford_theta(a_coulomb: float, rho: float, energy: float) -> float:
    return 2.0 * math.atan(a_coulomb / (2.0 * rho * energy))


def run_case(task: tuple[int, str, float, float, float, int, int]) -> dict[str, float | int | str]:
    condition, velocity_name, velocity_m_s, k0_b90, rho_over_b90, ures, dphires = task
    drag = make_drag(condition, vres=1, rhores=1, ures=ures, dphires=dphires)
    energy = 0.5 * drag.mu * velocity_m_s**2
    b90 = drag.A / (2.0 * energy)
    rho = rho_over_b90 * b90
    drag.k0 = k0_b90 / b90

    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            phi = float(drag.phiY(np.array([rho]), energy)[0])
    theta_yukawa = 2.0 * (drag.pi / 2.0 - phi)
    theta_rutherford = rutherford_theta(drag.A, rho, energy)
    return {
        "condition": condition,
        "velocity": velocity_name,
        "velocity_m_s": velocity_m_s,
        "k0_b90": k0_b90,
        "rho_over_b90": rho_over_b90,
        "theta_yukawa_rad": theta_yukawa,
        "theta_rutherford_rad": theta_rutherford,
        "abs_error_rad": abs(theta_yukawa - theta_rutherford),
        "rel_error": abs(theta_yukawa - theta_rutherford) / abs(theta_rutherford),
    }


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()

    tasks = []
    for condition in args.conditions:
        probe = make_drag(condition, args.vres, args.rhores, args.ures, args.dphires)
        for velocity_name, velocity_m_s in velocity_cases(probe).items():
            for rho_over_b90 in RHO_OVER_B90:
                for k0_b90 in K0_B90_VALUES:
                    tasks.append((condition, velocity_name, velocity_m_s, k0_b90, rho_over_b90, args.ures, args.dphires))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_case, tasks))

    rows.sort(key=lambda r: (r["condition"], r["velocity"], r["rho_over_b90"], r["k0_b90"]))
    write_csv(OUTDIR / "rutherford_limit.csv", rows)

    fig, axes = plt.subplots(len(args.conditions), 1, figsize=(7, 3 * len(args.conditions)), squeeze=False)
    for ax, condition in zip(axes[:, 0], args.conditions):
        subset = [row for row in rows if row["condition"] == condition]
        for velocity_name in sorted({row["velocity"] for row in subset}):
            group = [
                row
                for row in subset
                if row["velocity"] == velocity_name and row["rho_over_b90"] == 1.0
            ]
            ax.plot(
                [row["k0_b90"] for row in group],
                [row["theta_yukawa_rad"] for row in group],
                marker="o",
                label=f"Yukawa {velocity_name}",
            )
            if group:
                ax.axhline(
                    group[0]["theta_rutherford_rad"],
                    color=ax.lines[-1].get_color(),
                    linestyle="--",
                    linewidth=1,
                    label=f"Rutherford {velocity_name}",
                )
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("k0 b90")
        ax.set_ylabel("scattering angle [rad]")
        ax.set_title(condition_label(condition))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "rutherford_limit.png", dpi=200)


if __name__ == "__main__":
    freeze_support()
    main()
