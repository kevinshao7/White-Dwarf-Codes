import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from process_unforced import FitConfig, fit_decay, load_trajectory


class FitDecayTests(unittest.TestCase):
    def test_recovers_synthetic_exponential(self):
        rng = np.random.default_rng(42)
        time = np.linspace(0.0, 6.0, 100)
        amplitude, tau = 2.5e6, 20.0
        mean = amplitude * np.exp(-time / tau)
        velocities = mean[:, None] + rng.normal(0.0, 1.0e5, size=(len(time), 2000))
        result = fit_decay(
            velocities,
            time,
            condition=0,
            nominal_velocity=amplitude,
            source=Path("synthetic.np"),
            source_hash="synthetic",
            config=FitConfig(skip_rows=0),
        )
        self.assertNotEqual(result.status, "failed")
        self.assertAlmostEqual(result.amplitude / amplitude, 1.0, delta=0.03)
        self.assertAlmostEqual(result.tau / tau, 1.0, delta=0.12)

    def test_rejects_non_increasing_time(self):
        bad = np.array([[1.0, 2.0, 0.0], [0.9, 1.9, 0.0]])
        with patch("process_unforced.np.loadtxt", return_value=bad):
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                load_trajectory(Path("bad.np"))

    def test_ignores_start_end_uncertainty_overlap(self):
        rng = np.random.default_rng(7)
        time = np.linspace(0.0, 10.0, 80)
        one_snapshot = 1.0e6 + rng.normal(0.0, 2.0e5, size=2000)
        velocities = np.tile(one_snapshot, (len(time), 1))
        result = fit_decay(
            velocities,
            time,
            condition=1,
            nominal_velocity=1.0e6,
            source=Path("constant.np"),
            source_hash="synthetic",
            config=FitConfig(skip_rows=0),
        )
        self.assertEqual(result.status, "ignored")
        self.assertIn("start_end_velocity_intervals_overlap", result.quality_flags)

    def test_fit_start_uses_physical_time_not_index_distance(self):
        times = np.array([0.0, 1.0, 2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0])
        means = np.array([0.0, 8.0, 10.0, 7.0, 5.0, 3.5, 2.5, 1.8, 1.2]) * 1.0e5
        velocities = np.tile(means[:, None], (1, 100))
        result = fit_decay(
            velocities,
            times,
            condition=2,
            nominal_velocity=1.0e6,
            source=Path("uneven_times.np"),
            source_hash="synthetic",
            config=FitConfig(skip_rows=1),
        )
        # skip_rows=1 means row 0 is the last thermalization sample. Thus
        # thermalization ends at t=0, peak is t=2, target is t=4, and the first
        # recorded sample at or after that target is index 3 (t=5).
        self.assertEqual(result.thermalization_end_time, 0.0)
        self.assertEqual(result.target_fit_start_time, 4.0)
        self.assertEqual(result.start_index, 3)
        self.assertGreaterEqual(result.tau, 1.0e-20)


if __name__ == "__main__":
    unittest.main()
