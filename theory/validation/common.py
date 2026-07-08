from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import sys
import warnings
from pathlib import Path

import numpy as np

THEORY_DIR = Path(__file__).resolve().parents[1]
if str(THEORY_DIR) not in sys.path:
    sys.path.insert(0, str(THEORY_DIR))

from dragbase2 import DragFourth

CM_PER_S_TO_M_PER_S = 1.0e-2


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--conditions", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--vres", type=int, default=50)
    parser.add_argument("--rhores", type=int, default=180)
    parser.add_argument("--ures", type=int, default=180)
    parser.add_argument("--dphires", type=int, default=180)
    return parser


def make_drag(
    condition: int,
    vres: int = 50,
    rhores: int = 180,
    ures: int = 180,
    dphires: int = 180,
    cutoff_radius_factor: float = 1.0,
    vrel_sigma_width: float = 4.0,
    rhomax_fraction: float = 0.3,
) -> DragFourth:
    drag = DragFourth(
        condition,
        vres=vres,
        rhores=rhores,
        ures=ures,
        dphires=dphires,
        vrel_sigma_width=vrel_sigma_width,
        rhomax_fraction=rhomax_fraction,
    )
    if cutoff_radius_factor != 1.0:
        set_cutoff_radius_factor(drag, cutoff_radius_factor)
    return drag


def set_cutoff_radius_factor(drag: DragFourth, factor: float) -> None:
    radius = factor / drag.ustart
    drag.ustart = 1.0 / radius
    drag.E0Y = drag.A * np.exp(-drag.k0 / drag.ustart) * drag.ustart


def velocity_cases(drag: DragFourth) -> dict[str, float]:
    thermal = math.sqrt(drag.kb * drag.T / drag.mu)
    return {
        "low_1_cm_s": 1.0 * CM_PER_S_TO_M_PER_S,
        "thermal_1d": thermal,
        "high_2e7_cm_s": 2.0e7 * CM_PER_S_TO_M_PER_S,
    }


def condition_label(condition: int) -> str:
    drag = DragFourth(condition)
    return f"T={drag.T:.0e} K, rho={drag.gcc:.0e} g/cm^3"


def cutoff_defaults(condition: int) -> dict[str, float | str]:
    drag = DragFourth(condition)
    interparticle_spacing_m = 1.0 / drag.ustart
    return {
        "default_outer_radius_definition": "hydrogen interparticle spacing",
        "default_outer_radius_m": interparticle_spacing_m,
        "hydrogen_interparticle_spacing_m": interparticle_spacing_m,
        "electron_debye_radius_m": drag.lD,
        "default_rhomax_m": drag.rhomax_fraction * interparticle_spacing_m,
        "default_vrel_sigma_width": drag.vrel_sigma_width,
    }


def quiet_drag(drag: DragFourth, velocity_m_s: float) -> float:
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(drag.drag(velocity_m_s))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def relative_to_reference(values: list[float]) -> list[float]:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return [math.nan for _ in values]
    reference = finite_values[-1]
    if reference == 0.0:
        return [math.nan for _ in values]
    return [abs(value - reference) / abs(reference) for value in values]
