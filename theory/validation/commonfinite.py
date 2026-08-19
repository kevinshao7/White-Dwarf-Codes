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

try:
    from ..finite.finite_launch import DEFAULT_METHOD, METHODS, FiniteLaunchDrag
except ImportError:
    from theory.finite.finite_launch import DEFAULT_METHOD, METHODS, FiniteLaunchDrag

CM_PER_S_TO_M_PER_S = 1.0e-2
# r_i = launch radius in units of the hydrogen interparticle spacing a_H.
DEFAULT_CUTOFF_RADIUS_FACTOR = 1.0
# b_max = DEFAULT_RHOMAX_FRACTION * r_i.  1.0 covers the whole launch sphere.
DEFAULT_RHOMAX_FRACTION = 1.0


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--conditions", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--vres", type=int, default=201)
    parser.add_argument("--rhores", type=int, default=180)
    parser.add_argument("--ures", type=int, default=180)
    parser.add_argument("--dphires", type=int, default=180)
    parser.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD)
    parser.add_argument("--quad-epsabs", type=float, default=0.0)
    parser.add_argument("--quad-epsrel", type=float, default=1.0e-8)
    parser.add_argument("--quad-limit", type=int, default=200)
    return parser


def make_drag(
    condition: int,
    vres: int = 201,
    rhores: int = 180,
    ures: int = 180,
    dphires: int = 180,
    cutoff_radius_factor: float = DEFAULT_CUTOFF_RADIUS_FACTOR,
    vrel_sigma_width: float = 4.0,
    rhomax_fraction: float = DEFAULT_RHOMAX_FRACTION,
    dphi_endpoint_fraction: float = 1.0e-5,
    acipc: float = 1.0,
    method: str = DEFAULT_METHOD,
    quad_epsabs: float = 0.0,
    quad_epsrel: float = 1.0e-8,
    quad_limit: int = 200,
) -> FiniteLaunchDrag:
    """Construct the finite-launch drag solver.

    Particles start on a sphere of radius `r_i = cutoff_radius_factor * a_H`
    with relative speed `v_i`, conserved energy `E = mu v_i^2 / 2 + U(r_i)` and
    angular momentum `L = mu b v_i`, where `b` is the finite-launch impact
    parameter.  The scattering angle is `theta = pi - dphi - 2 alpha` with
    `sin(alpha) = b / r_i`, which vanishes identically for a free particle and
    reduces to `theta = pi - dphi` as `r_i -> infinity`.

    `cutoff_radius_factor` and `rhomax_fraction` are the two fit handles:
    they set `r_i` and `b_max = rhomax_fraction * r_i` respectively.

    `method` selects the quadrature (`"quad_quad"`, `"quad_angle"`,
    `"vectorized"`); the physics is identical in all three.
    """
    if acipc != 1.0:
        raise ValueError("acipc is fixed at 1 and is no longer a fit parameter")
    drag = FiniteLaunchDrag(
        condition,
        vres=vres,
        rhores=rhores,
        ures=ures,
        dphires=dphires,
        vrel_sigma_width=vrel_sigma_width,
        rhomax_fraction=rhomax_fraction,
        dphi_endpoint_fraction=dphi_endpoint_fraction,
        acipc=acipc,
        method=method,
        quad_epsabs=quad_epsabs,
        quad_epsrel=quad_epsrel,
        quad_limit=quad_limit,
    )
    if cutoff_radius_factor != 1.0:
        set_cutoff_radius_factor(drag, cutoff_radius_factor)
    return drag


def set_cutoff_radius_factor(drag: FiniteLaunchDrag, factor: float) -> None:
    """Set the launch radius to `factor * a_H` and refresh `U(r_i)`."""
    if factor <= 0.0:
        raise ValueError("cutoff_radius_factor must be positive")
    radius = factor / drag.ustart
    drag.ustart = 1.0 / radius
    drag.E0Y = drag.A * np.exp(-drag.k0 / drag.ustart) * drag.ustart


def velocity_cases(drag: FiniteLaunchDrag) -> dict[str, float]:
    thermal = math.sqrt(drag.kb * drag.T / drag.mu)
    return {
        "low_1_cm_s": 1.0 * CM_PER_S_TO_M_PER_S,
        "thermal_1d": thermal,
        "high_2e7_cm_s": 2.0e7 * CM_PER_S_TO_M_PER_S,
    }


def condition_label(condition: int) -> str:
    drag = FiniteLaunchDrag(condition)
    return f"T={drag.T:.0e} K, rho={drag.gcc:.0e} g/cm^3"


def cutoff_defaults(condition: int) -> dict[str, float | str]:
    drag = FiniteLaunchDrag(condition)
    interparticle_spacing_m = drag.launch_radius()
    set_cutoff_radius_factor(drag, DEFAULT_CUTOFF_RADIUS_FACTOR)
    return {
        "default_outer_radius_definition": "finite launch radius",
        "default_outer_radius_m": drag.launch_radius(),
        "hydrogen_interparticle_spacing_m": interparticle_spacing_m,
        "electron_debye_radius_m": drag.lD,
        "yukawa_screening_length_m": 1.0 / drag.k0,
        "default_rhomax_m": drag.launch_pmax(),
        "default_angle_cutoff_m": drag.launch_radius(),
        "default_acipc": drag.acipc,
        "default_vrel_sigma_width": drag.vrel_sigma_width,
        "default_method": drag.method,
    }


def quiet_drag(drag: FiniteLaunchDrag, velocity_m_s: float) -> float:
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
