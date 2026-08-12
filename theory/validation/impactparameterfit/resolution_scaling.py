from __future__ import annotations

import math


def scaled_impact_parameter_resolution(base_rhores: int, bmax_over_aH: float) -> int:
    """Keep launch-impact bin width roughly fixed as bmax/aH changes.

    The finite-launch grid is equally spaced in area, with edges proportional
    to sqrt(i / N).  At fixed physical p, local dp scales like pmax^2 / N, so
    keeping dp fixed requires N to scale as (bmax/aH)^2.
    """
    if base_rhores < 1:
        raise ValueError("base_rhores must be positive")
    if not math.isfinite(bmax_over_aH) or bmax_over_aH <= 0.0:
        raise ValueError("bmax/aH must be positive and finite")
    return max(2, int(math.ceil(base_rhores * bmax_over_aH**2)))


def scaled_scattering_angle_resolution(base_dphires: int, bmax_over_aH: float) -> int:
    """Scale the scattering-angle quadrature with cutoff length."""
    if base_dphires < 1:
        raise ValueError("base_dphires must be positive")
    if not math.isfinite(bmax_over_aH) or bmax_over_aH <= 0.0:
        raise ValueError("bmax/aH must be positive and finite")
    return max(2, int(math.ceil(base_dphires * bmax_over_aH)))


def scaled_resolution_for_bmax(resolution: dict[str, int], bmax_over_aH: float) -> dict[str, int]:
    scaled = dict(resolution)
    scaled["rhores"] = scaled_impact_parameter_resolution(int(resolution["rhores"]), bmax_over_aH)
    scaled["dphires"] = scaled_scattering_angle_resolution(int(resolution["dphires"]), bmax_over_aH)
    return scaled
