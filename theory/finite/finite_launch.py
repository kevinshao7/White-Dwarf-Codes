"""Finite-launch Yukawa drag on a silicon tracer in a hydrogen background.

Physical setup
--------------
A silicon tracer drifts with bulk velocity ``v_b`` through hydrogen with which
it is in thermal equilibrium.  Number and mass density of silicon are
negligible, ``n ~ n_H >> n_Si``.  Each binary Si-H collision is reduced to a
one-body problem in the centre-of-mass frame with reduced mass
``mu = m_H m_Si / (m_H + m_Si)`` moving in the screened Yukawa potential

    U(r) = A exp(-k_0 r) / r,   A = Z_H^eff Z_Si^eff q_e^2 / (4 pi eps_0),

with ``k_0 = 1 / lambda_S`` and the Melrose-corrected screening length
``lambda_S = lambda_D (1 + 2 T_f / 3 T)^(1/4)``.  All of that is inherited from
``DragFourth``.

Finite launch geometry
----------------------
Particles are *not* released from infinity.  They start on a sphere of radius
``r_i = launch_radius()`` with relative speed ``v_i`` and perpendicular offset
``b`` (the impact parameter of the straight line the particle would follow if
no force acted).  The conserved quantities are

    E = 1/2 mu v_i^2 + U(r_i),      L = mu b v_i,

so the orbital angle swept from launch, in to closest approach, and back out to
``r_i`` is

    Delta_phi = 2 Int_{r_min}^{r_i} dr (b v_i / r^2)
                / sqrt( (2/mu) (E - U(r) - mu b^2 v_i^2 / (2 r^2)) ).

The deflection of the velocity vector follows from the launch and arrival
geometry.  The velocity at launch points along ``phi_i + pi + alpha`` and at
arrival along ``phi_f - alpha``, with ``sin(alpha) = b / r_i``, hence

    theta = pi - Delta_phi - 2 alpha.

The two ``alpha`` terms (one per leg) are what make this reduce correctly:
for ``U = 0`` one has ``Delta_phi = pi - 2 alpha`` exactly, so ``theta = 0``,
and as ``r_i -> infinity`` (``alpha -> 0``) it collapses to the textbook
``theta = pi - Delta_phi``.

Regularised angle integral
--------------------------
Substituting ``u = 1/r`` and then ``u = u_0 - w t^2`` (with ``u_0`` the closest
approach, ``w = u_0 - u_i``) removes the inverse-square-root divergence at
``u_0``.  Writing the radial function as

    g(u) = 1 - U(1/u)/E - (rho u)^2,     rho = b v_i / v_inf = b v_i sqrt(mu/2E)

and using ``g(u_0) = 0`` to factor out ``(u_0 - u)`` analytically gives a
completely nonsingular integrand:

    Delta_phi = 4 rho sqrt(w) Int_0^1 dt / sqrt( Q(u_0 - w t^2) ),
    Q(u) = D(u)/E + rho^2 (u_0 + u),
    D(u) = (U(1/u_0) - U(1/u)) / (u_0 - u)   [divided difference].

No cancellation of nearly equal square roots is involved anywhere, and the
free-particle limit is reproduced analytically:
``Delta_phi = 4 arcsin(sqrt((1 - sin alpha)/2)) = pi - 2 alpha``.

Drag force
----------
    F = 2 pi n_H mu Int dv f(v) Int_0^{b_max} db b v |v| (1 - cos theta(v, b))

with ``f`` the 1D Maxwellian of the relative velocity, centred on ``v_b`` and
of width ``sqrt(k_B T / mu)``.  The ``v |v|`` (rather than ``v^2``) keeps the
integrand odd in the relative velocity, so the drag vanishes as ``v_b -> 0``;
for ``v_b >> c_H`` it is identical to the ``v_rel^2`` form of the derivation.
The velocity integral is a plain midpoint sum over speed with the positive and
negative Maxwellian lobes paired at equal ``|v|``, which avoids differencing
two large thermal lobes at low drift.

Quadrature variants
-------------------
``method`` selects how both inner integrals are evaluated. The physics is
identical; only the quadrature differs.

``"quad_quad"``
    ``scipy.integrate.quad`` on both the impact parameter and the scattering
    angle (nested adaptive quadrature). Reference accuracy, slowest -- used
    as the ground-truth value in ``theory/finite/convergence/``.
``"vectorized"``
    Log-spaced midpoint rule in ``b`` and a vectorised midpoint rule in ``t``,
    with a vectorised bisection for the closest approach. The production
    candidate: both grids converge as O(n^-2)
    (``theory/finite/convergence/``), and it benchmarks 40-90x faster than
    ``"quad_quad"`` at resolutions that already hold the combined error under
    1e-3 relative to it.

Two intermediate "mixed" schemes (adaptive quad on one integral, midpoint on
the other) existed during development to isolate each grid's discretization
error independently; both were dropped once ``"vectorized"`` was confirmed as
the fastest scheme that meets accuracy targets, since production code never
needs a scheme its own convergence study doesn't recommend. `rhores` and
`dphires` remain independent knobs on `"vectorized"` (see `orbit_angle` and
`impact_parameter_integral`), which is what the convergence study now uses in
their place: hold one axis at a resolution fine enough that its own error is
negligible and sweep the other, rather than making it exact via quadrature.

Why the impact parameter is log-spaced
--------------------------------------
An equal-area partition (``b_j^2 - b_{j-1}^2`` constant) is the natural choice
if the drag integrand is flat in area, which holds only at low drift.  As the
drift rises the integrand concentrates at small ``b``, and equal-area binning
puts its *coarsest* bin, ``[0, b_max/sqrt(rhores)]``, exactly on the peak:
measured against adaptive quad it errs by 4e-2 at condition 0 / v = 1e4 (where
25% of the integral lies inside that first bin) and by 6e-1 at v = 2e5, versus
4e-5 at v = 1e2.  Log spacing resolves the small-``b`` peak instead.  The grid
runs from ``bmin_fraction * b_max`` to ``b_max``; the remaining core
``[0, b_min]`` is added in closed form rather than dropped.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq

THEORY_DIR = Path(__file__).resolve().parents[1]
if str(THEORY_DIR) not in sys.path:
    sys.path.insert(0, str(THEORY_DIR))

from dragbase2 import DragFourth

METHODS = ("quad_quad", "vectorized")
DEFAULT_METHOD = "quad_quad"

# Inner cutoff of the log grid, as a fraction of b_max.
DEFAULT_BMIN_FRACTION = 1.0e-8

# Relative gap below which the divided difference D(u) is replaced by the
# midpoint derivative.  Above it the subtraction keeps ~10 significant digits.
_DIVIDED_DIFFERENCE_FLOOR = 1.0e-7


class FiniteLaunchDrag(DragFourth):
    """Yukawa drag with particles launched from a finite radius ``r_i``.

    Parameters mirror :class:`DragFourth` with three additions:

    ``method``
        One of ``METHODS``; see the module docstring.
    ``quad_epsabs`` / ``quad_epsrel`` / ``quad_limit``
        Tolerances handed to ``scipy.integrate.quad``.  Both inner integrals
        are rescaled to O(1) magnitude before integration, so ``epsrel``
        carries the tolerance and ``epsabs`` defaults to zero.
    ``bmin_fraction``
        Inner cutoff of the log-spaced impact-parameter grid, as a fraction of
        ``b_max``.  The ``[0, b_min]`` core is added in closed form.

    ``rhomax_fraction`` sets ``b_max = rhomax_fraction * r_i`` and defaults to
    1.0, i.e. the impact parameter covers the whole launch sphere.  ``b = r_i``
    is a tangent launch, where ``alpha = pi/2`` and ``theta -> 0`` smoothly, so
    nothing is truncated by hand.

    ``ures`` is accepted for call-signature compatibility and is unused: the
    substitution above removed the integral it used to resolve.
    """

    def __init__(
        self,
        conditions,
        vres: int = 201,
        rhores: int = 180,
        ures: int = 180,
        dphires: int = 180,
        vrel_sigma_width: float = 4.0,
        rhomax_fraction: float = 1.0,
        dphi_endpoint_fraction: float = 1.0e-5,
        acipc: float = 1.0,
        method: str = DEFAULT_METHOD,
        quad_epsabs: float = 0.0,
        quad_epsrel: float = 1.0e-8,
        quad_limit: int = 200,
        bmin_fraction: float = DEFAULT_BMIN_FRACTION,
    ):
        super().__init__(
            conditions,
            vres=vres,
            rhores=rhores,
            ures=ures,
            dphires=dphires,
            vrel_sigma_width=vrel_sigma_width,
            rhomax_fraction=rhomax_fraction,
            dphi_endpoint_fraction=dphi_endpoint_fraction,
            acipc=acipc,
        )
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if rhomax_fraction > 1.0:
            raise ValueError(
                "rhomax_fraction must not exceed 1: b > r_i has no launch geometry"
            )
        if not 0.0 < bmin_fraction < 1.0:
            raise ValueError("bmin_fraction must lie in (0, 1)")
        self.method = method
        self.quad_epsabs = float(quad_epsabs)
        self.quad_epsrel = float(quad_epsrel)
        self.quad_limit = int(quad_limit)
        self.bmin_fraction = float(bmin_fraction)
        self.quad_warnings = 0

    # ------------------------------------------------------------------
    # launch geometry
    # ------------------------------------------------------------------
    def launch_radius(self) -> float:
        """Radius ``r_i`` of the sphere the relative coordinate starts on."""
        return 1.0 / self.ustart

    def launch_pmax(self) -> float:
        """Impact-parameter ceiling ``b_max = rhomax_fraction * r_i``."""
        return self.rhomax_fraction * self.launch_radius()

    def launch_energy(self, speed: float) -> float:
        """``E = 1/2 mu v_i^2 + U(r_i)``, the conserved two-body energy."""
        return 0.5 * self.mu * speed**2 + self.E0Y

    def asymptotic_rho(self, b, speed: float, energy: float):
        """``rho = b v_i / v_inf``: the impact parameter the orbit would have
        had if released from infinity with the same ``E`` and ``L``."""
        v_inf = math.sqrt(2.0 * energy / self.mu)
        return np.asarray(b, dtype=np.float64) * abs(speed) / v_inf

    def launch_alpha(self, b):
        """``alpha = arcsin(b / r_i)``, the launch angle of the velocity to
        the inward radial direction."""
        return np.arcsin(np.clip(np.asarray(b, dtype=np.float64) * self.ustart, 0.0, 1.0))

    # backwards-compatible aliases used by the impactparameterfit scripts
    def launch_p_to_orbit_rho(self, p, v_start: float, energy: float):
        return self.asymptotic_rho(p, v_start, energy)

    def orbit_rho_to_launch_p(self, rho, v_start: float, energy: float):
        v_inf = math.sqrt(2.0 * energy / self.mu)
        return np.asarray(rho, dtype=np.float64) * v_inf / abs(v_start)

    # ------------------------------------------------------------------
    # radial function and closest approach
    # ------------------------------------------------------------------
    def yukawa_u(self, u):
        """``U(1/u) = A u exp(-k_0 / u)``."""
        u = np.asarray(u, dtype=np.float64)
        with np.errstate(over="ignore", divide="ignore"):
            return self.A * u * np.exp(-self.k0 / u)

    def radial_g(self, u, rho, energy: float):
        """``g(u) = 1 - U(1/u)/E - (rho u)^2``; zero at closest approach."""
        return 1.0 - self.yukawa_u(u) / energy - np.square(np.asarray(rho, dtype=np.float64) * u)

    def closest_approach_u(self, rho: float, energy: float) -> float:
        """Scalar root ``u_0 > u_i`` of :meth:`radial_g` via ``brentq``.

        ``g`` is strictly decreasing for a repulsive Yukawa (``A > 0``), so the
        root is unique.  ``g(u_i) = (1/2) mu v_i^2 (1 - (b/r_i)^2) / E >= 0``
        guarantees the lower bracket; ``g -> -inf`` guarantees the upper one.
        """
        lower = self.ustart
        if self.radial_g(lower, rho, energy) <= 0.0:
            return lower
        upper = 2.0 * lower
        for _ in range(400):
            if self.radial_g(upper, rho, energy) < 0.0:
                break
            upper *= 2.0
        else:
            raise FloatingPointError("closest approach: failed to bracket the root")
        return float(brentq(self.radial_g, lower, upper, args=(rho, energy), maxiter=200))

    def closest_approach_u_array(self, rho, energy: float, iterations: int = 100) -> np.ndarray:
        """Vectorised bisection counterpart of :meth:`closest_approach_u`."""
        rho = np.asarray(rho, dtype=np.float64)
        lower = np.full(rho.shape, self.ustart, dtype=np.float64)
        no_penetration = self.radial_g(lower, rho, energy) <= 0.0
        upper = 2.0 * lower
        for _ in range(400):
            needs_growth = self.radial_g(upper, rho, energy) >= 0.0
            if not np.any(needs_growth):
                break
            upper = np.where(needs_growth, upper * 2.0, upper)
        else:
            raise FloatingPointError("closest approach: failed to bracket the root")
        for _ in range(iterations):
            middle = 0.5 * (lower + upper)
            positive = self.radial_g(middle, rho, energy) > 0.0
            lower = np.where(positive, middle, lower)
            upper = np.where(positive, upper, middle)
        return np.where(no_penetration, self.ustart, 0.5 * (lower + upper))

    # ------------------------------------------------------------------
    # orbital angle
    # ------------------------------------------------------------------
    def _divided_difference(self, u, u0, width_scale):
        """``D(u) = (U(1/u_0) - U(1/u)) / (u_0 - u)``, cancellation-guarded.

        Within ``_DIVIDED_DIFFERENCE_FLOOR`` of ``u_0`` the subtraction loses
        too many digits, so the midpoint derivative ``dU/du`` is used instead;
        it agrees to ``O((u_0 - u)^2)``.
        """
        u = np.asarray(u, dtype=np.float64)
        gap = u0 - u
        # `<=` also catches the degenerate tangent launch b = r_i, where the
        # closest approach coincides with the launch radius and gap == 0.
        near = gap <= _DIVIDED_DIFFERENCE_FLOOR * width_scale
        safe_gap = np.where(near, 1.0, gap)
        difference = (self.yukawa_u(u0) - self.yukawa_u(u)) / safe_gap
        middle = np.where(near, 0.5 * (u + u0), u0)
        with np.errstate(over="ignore", divide="ignore"):
            derivative = self.A * np.exp(-self.k0 / middle) * (1.0 + self.k0 / middle)
        return np.where(near, derivative, difference)

    def _angle_kernel(self, t, u0: float, width: float, rho: float, energy: float):
        """``1 / sqrt(Q(u_0 - w t^2))``, the regularised angle integrand."""
        t = np.asarray(t, dtype=np.float64)
        u = u0 - width * np.square(t)
        q = self._divided_difference(u, u0, width) / energy + (rho**2) * (u0 + u)
        return 1.0 / np.sqrt(q)

    def orbit_angle(self, b, speed: float, energy: float, use_quad: bool | None = None):
        """Total swept angle ``Delta_phi`` from ``r_i`` in to ``r_min`` and back.

        Vectorised in ``b``.  Returns 0 for ``b >= r_i`` (tangent launch, the
        particle never moves inward).  ``use_quad`` overrides the instance
        method (``True`` for adaptive quadrature, ``False`` for the
        vectorised midpoint rule over ``dphires`` nodes).
        """
        if use_quad is None:
            use_quad = self.method == "quad_quad"
        b = np.atleast_1d(np.asarray(b, dtype=np.float64))
        rho = self.asymptotic_rho(b, speed, energy)

        if not use_quad:
            u0 = self.closest_approach_u_array(rho, energy)
            width = u0 - self.ustart
            nodes = (np.arange(self.dphires, dtype=np.float64) + 0.5) / self.dphires
            kernel = self._angle_kernel(
                nodes[None, :], u0[:, None], width[:, None], rho[:, None], energy
            )
            integral = kernel.mean(axis=1)
            return np.where(width > 0.0, 4.0 * rho * np.sqrt(np.maximum(width, 0.0)) * integral, 0.0)

        flat_rho = rho.ravel()
        angles = np.zeros(flat_rho.size, dtype=np.float64)
        for index in range(flat_rho.size):
            rho_i = float(flat_rho[index])
            u0 = self.closest_approach_u(rho_i, energy)
            width = u0 - self.ustart
            if width <= 0.0:
                continue
            integral = self._quad(
                lambda t, u0=u0, width=width, rho_i=rho_i: float(
                    self._angle_kernel(t, u0, width, rho_i, energy)
                ),
                0.0,
                1.0,
            )
            angles[index] = 4.0 * rho_i * math.sqrt(width) * integral
        return angles.reshape(b.shape)

    def scattering_angle(self, b, speed: float, energy: float | None = None):
        """``theta = pi - Delta_phi - 2 alpha``, vectorised in ``b``.

        ``math.pi`` rather than the inherited ``self.pi``: the latter is
        truncated to 3.1415926535, and a 9e-11 rad absolute bias would swamp
        the genuinely small deflections at large ``b`` that dominate the drag.
        """
        if energy is None:
            energy = self.launch_energy(speed)
        return math.pi - self.orbit_angle(b, speed, energy) - 2.0 * self.launch_alpha(b)

    def finite_scattering_half_angle(self, p, v_start: float, energy: float):
        """``theta / 2``.  Kept for callers that want ``2 sin^2(theta/2)``."""
        return 0.5 * self.scattering_angle(p, v_start, energy)

    def momentum_transfer_factor(self, b, speed: float, energy: float):
        """``1 - cos(theta)``, the fraction of ``mu v_rel`` transferred."""
        return 1.0 - np.cos(self.scattering_angle(b, speed, energy))

    # ------------------------------------------------------------------
    # impact-parameter integral
    # ------------------------------------------------------------------
    def launch_pmin(self) -> float:
        """Inner cutoff ``b_min`` of the log-spaced impact-parameter grid."""
        return self.bmin_fraction * self.launch_pmax()

    def _finite_launch_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Log-spaced midpoints in ``b``, from ``b_min`` to ``b_max``.

        In ``y = ln(b)`` the integral is ``Int b^2 (1 - cos theta) dy``, so the
        effective width returned is ``b * dy``: callers keep summing
        ``b * db * f`` unchanged, and ``b * (b dy)`` is the correct measure.
        """
        bmax = self.launch_pmax()
        bmin = self.launch_pmin()
        if self.rhores < 1 or bmax <= 0.0 or bmin <= 0.0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        span = math.log(bmax) - math.log(bmin)
        dy = span / self.rhores
        y = math.log(bmin) + (np.arange(self.rhores, dtype=np.float64) + 0.5) * dy
        centers = np.exp(y)
        return centers, centers * dy

    def _core_integral(self, speed: float, energy: float) -> float:
        """``Int_0^{b_min} db b (1 - cos theta)``, the part the log grid omits.

        Deflection is essentially head-on across this core, so the factor is
        taken at ``b_min``: the contribution is ``(1 - cos theta) b_min^2 / 2``,
        of order ``bmin_fraction^2`` relative to the whole.  Negligible at the
        default 1e-8, but accounted for rather than silently truncated.
        """
        bmin = self.launch_pmin()
        if bmin <= 0.0:
            return 0.0
        factor = float(np.ravel(self.momentum_transfer_factor(bmin, speed, energy))[0])
        return 0.5 * factor * bmin**2

    def impact_parameter_integral(self, speed: float, energy: float) -> float:
        """``Int_0^{b_max} db b (1 - cos theta(b))``, in m^2."""
        bmax = self.launch_pmax()
        bmin = self.launch_pmin()
        if bmax <= 0.0 or bmin <= 0.0:
            return 0.0

        if self.method == "quad_quad":
            # Adaptive quad in y = ln(b/b_max) for the same reason the grid is
            # log-spaced, and scaled by b_max^2 so the integrand is O(1) and
            # the relative tolerance is meaningful.
            def integrand(y: float) -> float:
                scale = math.exp(y)
                factor = self.momentum_transfer_factor(scale * bmax, speed, energy)
                return scale**2 * float(np.ravel(factor)[0])

            value = self._quad(integrand, math.log(self.bmin_fraction), 0.0)
            return bmax**2 * value + self._core_integral(speed, energy)

        b, db = self._finite_launch_grid()
        if len(b) == 0:
            return 0.0
        factor = self.momentum_transfer_factor(b, speed, energy)
        valid = np.isfinite(factor)
        if not np.any(valid):
            return 0.0
        return float(np.sum(b[valid] * db[valid] * factor[valid])) + self._core_integral(
            speed, energy
        )

    def _quad(self, integrand, lower: float, upper: float) -> float:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IntegrationWarning)
            value, _ = quad(
                integrand,
                lower,
                upper,
                epsabs=self.quad_epsabs,
                epsrel=self.quad_epsrel,
                limit=self.quad_limit,
            )
        self.quad_warnings += sum(
            1 for entry in caught if issubclass(entry.category, IntegrationWarning)
        )
        return float(value)

    # ------------------------------------------------------------------
    # drag force
    # ------------------------------------------------------------------
    def _drag_speed_integrand(self, speed: float, weight: float) -> float:
        """Inner ``b`` integral weighted by flux and Maxwellian at one speed."""
        if speed <= 0.0 or weight == 0.0:
            return 0.0
        energy = self.launch_energy(speed)
        return speed**2 * weight * self.impact_parameter_integral(speed, energy)

    def drag(self, vb: float) -> float:
        """Drag force on the silicon tracer, in newtons, for bulk speed ``vb``.

        Basic midpoint rule over the relative speed.  The positive and negative
        Maxwellian lobes are evaluated at the same ``|v|`` and subtracted there,
        so the integrand is exactly odd in the relative velocity: the drag
        vanishes at ``vb = 0`` by construction rather than by cancellation of
        two large numbers.
        """
        sigma_v = math.sqrt(self.kb * self.T / self.mu)
        width = self.vrel_sigma_width * sigma_v
        vmin = vb - width
        vmax = vb + width
        speed_min = 0.0 if vmin <= 0.0 <= vmax else min(abs(vmin), abs(vmax))
        speed_max = max(abs(vmin), abs(vmax))
        if self.vres < 1 or speed_max <= speed_min:
            return 0.0

        ds = (speed_max - speed_min) / self.vres
        speeds = speed_min + (np.arange(self.vres, dtype=np.float64) + 0.5) * ds
        norm = math.sqrt(self.mu / (2.0 * math.pi * self.kb * self.T))

        total = 0.0
        for speed in speeds:
            positive = 0.0
            negative = 0.0
            if vmin <= speed <= vmax:
                positive = norm * math.exp(-self.mu * (speed - vb) ** 2 / (2.0 * self.kb * self.T))
            if vmin <= -speed <= vmax:
                negative = norm * math.exp(-self.mu * (-speed - vb) ** 2 / (2.0 * self.kb * self.T))
            total += self._drag_speed_integrand(float(speed), positive - negative)
        return 2.0 * math.pi * self.nh * self.mu * total * ds


if __name__ == "__main__":
    vb = 1.0e4
    for name in METHODS:
        forces = [FiniteLaunchDrag(condition, method=name).drag(vb) for condition in range(4)]
        print(f"{name:>11}: " + "  ".join(f"{value:.6e}" for value in forces))
