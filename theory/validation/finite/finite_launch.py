from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

THEORY_DIR = Path(__file__).resolve().parents[2]
if str(THEORY_DIR) not in sys.path:
    sys.path.insert(0, str(THEORY_DIR))

from dragbase2 import DragFourth


class FiniteLaunchDrag(DragFourth):
    """Drag solver for particles launched from a finite spherical boundary.

    The inherited orbit equations use the asymptotic impact parameter `rho`,
    with angular momentum `L = mu * rho * v_inf`.  At a finite launch radius
    `r_start`, the conserved angular momentum is instead
    `L = mu * r_start * v_start * sin(theta)`.  The finite perpendicular
    launch coordinate is therefore

        p = r_start * sin(theta)
        rho = p * v_start / v_inf

    where `theta` is the angle between the incoming velocity and the inward
    radial direction.
    """

    def launch_radius(self) -> float:
        return 1.0 / self.ustart

    def launch_pmax(self) -> float:
        return self.rhomax_fraction * self.launch_radius()

    def launch_p_to_orbit_rho(self, p: np.ndarray, v_start: float, energy: float) -> np.ndarray:
        v_inf = math.sqrt(2.0 * energy / self.mu)
        return p * abs(v_start) / v_inf

    def orbit_rho_to_launch_p(self, rho: np.ndarray, v_start: float, energy: float) -> np.ndarray:
        v_inf = math.sqrt(2.0 * energy / self.mu)
        return rho * v_inf / abs(v_start)

    def _finite_yukawa_angle_from_launch(self, rho: np.ndarray, energy: float, u0: np.ndarray) -> np.ndarray:
        """Integrate the Yukawa orbital angle from `r_start` to closest approach."""
        results = np.zeros(len(rho), dtype=np.float64)
        lower = self.ustart
        for i, rhoi in enumerate(rho):
            upper = u0[i]
            if upper <= lower:
                continue
            x = (np.arange(self.dphires, dtype=np.float64) + 0.5) / self.dphires
            uarr = upper - (upper - lower) * np.square(x)
            dudx = 2.0 * (upper - lower) * x
            results[i] = np.nansum(self.Yint(uarr, rhoi, energy) * dudx) / self.dphires
        return results

    def finite_scattering_half_angle(self, p: np.ndarray, v_start: float, energy: float) -> np.ndarray:
        """Return half the finite-start deflection angle.

        For zero interaction this returns exactly zero: the finite free angle
        from launch radius to closest approach is `pi/2 - theta`, and the
        Yukawa angle is compared against that finite free reference.
        """
        r_start = self.launch_radius()
        sin_theta = np.clip(p / r_start, 0.0, 1.0)
        theta = np.arcsin(sin_theta)
        finite_free_angle = self.pi / 2.0 - theta

        rho = self.launch_p_to_orbit_rho(p, v_start, energy)
        u0 = self.umax(rho, energy)
        finite_yukawa_angle = self._finite_yukawa_angle_from_launch(rho, energy, u0)
        return finite_free_angle - finite_yukawa_angle

    def _finite_launch_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Midpoint grid in perpendicular launch coordinate with equal area bins."""
        pmax = self.launch_pmax()
        if self.rhores < 1 or pmax <= 0.0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        edges = pmax * np.sqrt(np.linspace(0.0, 1.0, self.rhores + 1, dtype=np.float64))
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        return centers, widths

    def _drag_speed_integrand(self, speed: float, weight: float) -> float:
        if speed <= 0.0 or weight == 0.0:
            return 0.0

        energy = 0.5 * self.mu * speed**2 + self.E0Y
        p, dp = self._finite_launch_grid()
        if len(p) == 0:
            return 0.0

        half_theta = self.finite_scattering_half_angle(p, speed, energy)
        valid = np.isfinite(half_theta)
        if not np.any(valid):
            return 0.0

        return float(
            np.sum(
                p[valid]
                * dp[valid]
                * speed**2
                * weight
                * (2.0 * np.square(np.sin(half_theta[valid])))
            )
        )
