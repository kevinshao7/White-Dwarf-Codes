"""Shared CLI/data-loading helpers for the velocity-dependent b_max fits in
this directory.

Both fit scripts here (`fit_bmax_per_point.py`, `fit_bmax_two_regimes.py`)
build directly on `theory/finite/lammps_fit/fit_bmax_to_lammps.py` --
LAMMPS/DAIS data loading, point filtering/selection, and the single-drag-
evaluation plumbing -- rather than reimplementing it. That script fits one
constant `b_max/a_H` per condition; the two scripts here ask whether that
constant is really constant across velocity, in two different ways. Import
it as a flat module (`fit_bmax_to_lammps`, not a `finite.lammps_fit.*`
package import) because `lammps_fit/` has no `__init__.py`, mirroring how
`finite_launch.py` itself flat-imports `dragbase2`.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

LAMMPS_FIT_DIR = Path(__file__).resolve().parents[1] / "lammps_fit"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(LAMMPS_FIT_DIR) not in sys.path:
    sys.path.insert(0, str(LAMMPS_FIT_DIR))

import fit_bmax_to_lammps as base  # noqa: E402

DataPoint = base.DataPoint
ALL_CONDITIONS = base.ALL_CONDITIONS
DAIS_CONDITIONS = base.DAIS_CONDITIONS
WEAKLY_COUPLED_CONDITIONS = base.WEAKLY_COUPLED_CONDITIONS


def thermal_velocity_cm_s(condition: int) -> float:
    """Relative-velocity thermal width ``v_th = sqrt(kB T / mu)``, in cm/s.

    The same quantity ``FiniteLaunchDrag.drag`` calls ``sigma_v``. It is
    fixed by ``condition`` alone (``DragFourth.T``, ``.mu``), independent of
    any fit parameter, so a bare ``DragFourth`` instance is enough -- no
    need to build a ``FiniteLaunchDrag``.
    """
    drag = base.DragFourth(condition)
    v_th_m_s = math.sqrt(drag.kb * drag.T / drag.mu)
    return v_th_m_s / base.CM_PER_S_TO_M_PER_S


def build_common_parser(description: str | None) -> argparse.ArgumentParser:
    """Argparse options shared by both scripts: data source, point
    filtering, drag-model bounds/resolution, and worker-pool settings.
    Each script adds its own point-selection and fit-specific arguments on
    top of this.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--conditions", nargs="+", type=int, default=list(ALL_CONDITIONS))
    parser.add_argument(
        "--lammps-results", type=Path,
        default=REPO_ROOT / "theory" / "dataprocessing" / "output" / "results.npy",
    )
    parser.add_argument(
        "--dais-results", type=Path,
        default=REPO_ROOT / "theory" / "dataprocessing" / "output_dais" / "results.npy",
    )
    parser.add_argument("--samples-per-fit", type=int, default=20)
    parser.add_argument("--min-velocity-cm-s", type=float, default=1.0e2)
    parser.add_argument("--max-velocity-cm-s", type=float, default=1.0e8)
    parser.add_argument("--max-relative-sigma", type=float, default=0.5)
    parser.add_argument("--bmax-min", type=float, default=base.DEFAULT_BMAX_MIN)
    parser.add_argument("--bmax-max", type=float, default=base.DEFAULT_BMAX_MAX)
    parser.add_argument("--method", choices=("quad_quad", "vectorized"), default="vectorized")
    parser.add_argument(
        "--resolution", type=int, default=base.DEFAULT_RESOLUTION,
        help="rhores = dphires used for every drag evaluation during the fit",
    )
    parser.add_argument("--vres", type=int, default=base.DEFAULT_VRES)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--gpu-devices", type=str, default=None,
        help=(
            "Comma-separated CUDA device ids (e.g. '0,1') to batch drag evaluations across "
            "via FiniteLaunchDrag.drag_batch(xp=cupy) instead of the CPU --workers process "
            "pool. Requires cupy; see fit_bmax_to_lammps.run_fit_points_gpu's docstring for "
            "the GPU-untested caveat."
        ),
    )
    parser.add_argument(
        "--heartbeat-seconds", type=float, default=12.0,
        help="Print a status line at least this often even if no task has finished yet.",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-evaluation progress printing")
    return parser


def load_and_filter_points(args: argparse.Namespace) -> tuple[list[DataPoint], list[DataPoint], set[int]]:
    """Validate args, load LAMMPS(+DAIS) points, and apply the velocity/
    relative-sigma filter. Returns ``(all_points, filtered_points, conditions)``;
    callers do their own point *selection* (quantile grouping, regime
    splitting) on top of ``filtered_points``.
    """
    if not args.lammps_results.exists():
        raise SystemExit(f"--lammps-results not found: {args.lammps_results}")

    conditions = set(args.conditions)
    if conditions.intersection(DAIS_CONDITIONS) and not args.dais_results.exists():
        raise SystemExit(f"--dais-results not found: {args.dais_results}")

    all_points = base.load_all_points(args.lammps_results, args.dais_results, conditions, args.samples_per_fit)
    filtered = base.filter_points(all_points, args.min_velocity_cm_s, args.max_velocity_cm_s, args.max_relative_sigma)
    return all_points, filtered, conditions
