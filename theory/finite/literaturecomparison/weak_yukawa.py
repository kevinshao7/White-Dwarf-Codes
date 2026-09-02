"""Analytic, weak-coupling Yukawa drag models used for comparison.

The models here deliberately contain no fitted coefficients.  They are the
Maxwellian Landau/Spitzer dynamical-friction result (as presented, for
example, by Gurnett & Bhattacharjee and Boyd & Sanderson), with the Coulomb
logarithm regularised by the *same Yukawa screening length* as the finite
launch calculation.  They are not effective-potential or strong-coupling
fits, and should only be interpreted where ``b_90 / lambda_s << 1``.

Two analytic screening choices are provided:

``landau``
    Gurnett--Bhattacharjee / Boyd--Sanderson Maxwellian friction with
    ``0.5 log(1 + (lambda_s/b_min)^2)``.  This is the usual weak-scattering
    Coulomb-log prescription with a smooth finite cut-off.
``born_transport``
    Yukawa first-Born momentum-transfer theory, with the Li--Petrasso
    quantum lower cutoff.  Its generalized Coulomb logarithm is
    ``0.5 [log(1+x) - x/(1+x)]``.

In both cases ``b_min = sqrt(b_90^2 + b_q^2)`` uses the classical 90-degree
impact parameter and the Li--Petrasso diffraction scale
``b_q = hbar/(2 mu v)``.  Thus every input is a plasma property or a physical
constant, not a fitted parameter.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erf


MODEL_LABELS = {
    "landau": "Gurnett--Bhattacharjee / Boyd--Sanderson",
    "born_transport": "Yukawa Born transport (Li--Petrasso cutoff)",
}


def _positive_speed(speed_m_s):
    return np.maximum(np.asarray(speed_m_s, dtype=np.float64), np.finfo(np.float64).tiny)


def maxwell_drag_kernel(speed_m_s, background_mass: float, temperature_k: float, boltzmann: float):
    """Return ``erf(x) - 2*x*exp(-x^2)/sqrt(pi)`` for a Maxwellian background."""
    speed = _positive_speed(speed_m_s)
    x = speed * math.sqrt(background_mass / (2.0 * boltzmann * temperature_k))
    return erf(x) - 2.0 * x * np.exp(-np.square(x)) / math.sqrt(math.pi)


def cutoff_lengths(source, speed_m_s, screening_length_m: float | None = None):
    """Return ``(b_90, b_quantum, b_min, lambda_s)`` in metres.

    ``source`` is a :class:`finite.finite_launch.FiniteLaunchDrag` instance.
    Its Yukawa strength ``A``, reduced mass, and Melrose-corrected screening
    length are used directly so both calculations describe the same pair
    potential.
    """
    speed = _positive_speed(speed_m_s)
    lambda_s = float(1.0 / source.k0 if screening_length_m is None else screening_length_m)
    b_90 = source.A / (source.mu * np.square(speed))
    b_quantum = source.hbar / (2.0 * source.mu * speed)
    b_min = np.hypot(b_90, b_quantum)
    return b_90, b_quantum, b_min, lambda_s


def generalized_coulomb_logarithm(source, speed_m_s, model: str = "born_transport"):
    """Analytic weak-Yukawa generalized Coulomb logarithm.

    The result is dimensionless and is evaluated at the projectile/background
    relative speed represented by ``speed_m_s``.
    """
    if model not in MODEL_LABELS:
        raise ValueError(f"model must be one of {tuple(MODEL_LABELS)}, got {model!r}")
    _, _, b_min, lambda_s = cutoff_lengths(source, speed_m_s)
    x = np.square(lambda_s / b_min)
    if model == "landau":
        return 0.5 * np.log1p(x)
    return 0.5 * (np.log1p(x) - x / (1.0 + x))


def drag_force(source, speed_m_s, model: str = "born_transport"):
    """Magnitude of the analytic dynamical-friction force in newtons.

    This is the Gurnett--Bhattacharjee / Boyd--Sanderson Maxwellian
    dynamical-friction formula for a Si test particle in the hydrogen
    background represented by ``source``.  The returned quantity is positive;
    the physical force is opposite the supplied bulk-velocity vector.
    """
    speed = _positive_speed(speed_m_s)
    kernel = maxwell_drag_kernel(speed, source.mh, source.T, source.kb)
    coulomb_log = generalized_coulomb_logarithm(source, speed, model=model)
    # 4*pi*A^2/mu == (q_Si q_H)^2/(4*pi*eps0^2*mu), matching the
    # force (rather than acceleration) version of the literature formula.
    force = 4.0 * math.pi * source.nh * source.A**2 / source.mu * coulomb_log * kernel / np.square(speed)
    return np.asarray(force, dtype=np.float64)


def weak_coupling_ratio(source, speed_m_s):
    """Return ``b_90/lambda_s``; values much smaller than one are required."""
    b_90, _, _, lambda_s = cutoff_lengths(source, speed_m_s)
    return b_90 / lambda_s
