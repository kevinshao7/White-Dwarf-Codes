# Run from repository root:
#   python .\theory\finite\shape\plot_drag_curve.py
"""Drag-force-vs-velocity curves for FiniteLaunchDrag, all 4 conditions.

One 2x2 figure, one panel per condition, each showing |drag force| (N) vs
bulk velocity v_b (cm/s), log-log, over v_b in [1e4, 1e8] cm/s. A vertical
dashed line marks the hydrogen thermal speed

    v_H = sqrt(3 kB T / m_H)

(the 3D rms speed of a Maxwellian hydrogen background at the condition's
temperature T), using ``m_H`` = ``FiniteLaunchDrag.mh`` (per-atom hydrogen
mass, matching the mass used to build the Maxwellian relative-velocity
distribution in ``drag()``).

Uses ``method="vectorized"`` at the class default resolution (vres=30,
rhores=100, dphires=1000) -- the production candidate documented in
``finite_launch.py`` -- and ``drag_batch`` to evaluate all velocities for a
condition in one vectorised call.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUTDIR / ".matplotlib"))

import numpy as np

THEORY_DIR = Path(__file__).resolve().parents[2]
if str(THEORY_DIR) not in sys.path:
    sys.path.insert(0, str(THEORY_DIR))

from finite.finite_launch import FiniteLaunchDrag  # noqa: E402

CONDITIONS = (0, 1, 2, 3)
N_VELOCITIES = 40
VELOCITY_MIN_CM_S = 1.0e4
VELOCITY_MAX_CM_S = 1.0e8
CM_PER_M = 100.0


def condition_label(drag: FiniteLaunchDrag) -> str:
    return f"Condition {drag.condition_index}: T={drag.T:.0e} K, rho={drag.gcc:.0e} g/cm^3"


def main() -> None:
    import matplotlib.pyplot as plt

    velocities_cm_s = np.logspace(math.log10(VELOCITY_MIN_CM_S), math.log10(VELOCITY_MAX_CM_S), N_VELOCITIES)
    velocities_m_s = velocities_cm_s / CM_PER_M

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for condition, ax in zip(CONDITIONS, axes.ravel()):
        drag = FiniteLaunchDrag(condition, method="vectorized")
        drag.condition_index = condition
        forces_n = np.abs(np.asarray(drag.drag_batch(velocities_m_s)))

        thermal_speed_m_s = math.sqrt(3.0 * drag.kb * drag.T / drag.mh)
        thermal_speed_cm_s = thermal_speed_m_s * CM_PER_M

        valid = np.isfinite(forces_n) & (forces_n > 0.0)
        ax.plot(velocities_cm_s[valid], forces_n[valid], marker="o", markersize=3, linewidth=1.5, color="tab:blue")
        ax.axvline(
            thermal_speed_cm_s,
            color="k",
            linestyle="--",
            linewidth=1.2,
            label=f"hydrogen thermal speed\n$v_H=\\sqrt{{3k_BT/m_H}}$={thermal_speed_cm_s:.2e} cm/s",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("bulk velocity $v_b$ (cm/s)")
        ax.set_ylabel("|drag force| $F$ (N)")
        ax.set_title(condition_label(drag))
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Finite-launch Yukawa drag vs bulk velocity (method=\"vectorized\")", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    outfile = OUTDIR / "drag_curve.png"
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"Wrote {outfile}")


if __name__ == "__main__":
    main()
