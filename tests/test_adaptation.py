"""Regression tests for the online recalibration decision rules."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_online_adaptation import (  # noqa: E402
    Config,
    build_calibration_curve,
    confidence_recalibration_step,
    run_scenario,
)
from run_cost_aware_policy import (  # noqa: E402
    UtilitySurface,
    cost_aware_recalibration_step,
)


class AdaptationTests(unittest.TestCase):
    @staticmethod
    def calibration() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "measurement_error_rate": [0.005, 0.020],
                "detector_firing_rate": [0.10, 0.20],
            }
        )

    def test_confidence_gate_does_not_update_inside_interval(self) -> None:
        next_q, updated, alpha = confidence_recalibration_step(
            current_q=0.005,
            q_hat=0.010,
            q_low=0.008,
            q_high=0.012,
            detector_firing_rate=0.101,
            detector_firing_rate_se=0.01,
            calibration=self.calibration(),
            z=2.0,
            min_update_delta=0.0005,
        )
        self.assertFalse(updated)
        self.assertEqual(next_q, 0.005)
        self.assertEqual(alpha, 0.0)

    def test_confidence_gate_shrinks_toward_estimate(self) -> None:
        current_q = 0.005
        q_hat = 0.020
        next_q, updated, alpha = confidence_recalibration_step(
            current_q=current_q,
            q_hat=q_hat,
            q_low=0.018,
            q_high=0.022,
            detector_firing_rate=0.20,
            detector_firing_rate_se=0.001,
            calibration=self.calibration(),
            z=2.0,
            min_update_delta=0.0005,
        )
        self.assertTrue(updated)
        self.assertGreater(next_q, current_q)
        self.assertLess(next_q, q_hat)
        self.assertGreaterEqual(alpha, 0.2)
        self.assertLessEqual(alpha, 0.9)

    def test_cost_gate_respects_update_cost(self) -> None:
        grid = np.array([0.005, 0.020])
        utility = UtilitySurface(
            grid,
            grid,
            np.array([[0.01, 0.03], [0.04, 0.01]]),
        )
        next_q, updated, predicted = cost_aware_recalibration_step(
            current_q=0.005,
            candidate_q=0.020,
            q_low=0.019,
            q_high=0.020,
            utility=utility,
            future_shots=1_000,
            update_cost=5.0,
            min_update_delta=0.0005,
        )
        self.assertTrue(updated)
        self.assertEqual(next_q, 0.020)
        self.assertGreater(predicted, 5.0)

        next_q, updated, _ = cost_aware_recalibration_step(
            current_q=0.005,
            candidate_q=0.020,
            q_low=0.019,
            q_high=0.020,
            utility=utility,
            future_shots=1_000,
            update_cost=100.0,
            min_update_delta=0.0005,
        )
        self.assertFalse(updated)
        self.assertEqual(next_q, 0.005)

    def test_future_schedule_does_not_change_past_results(self) -> None:
        config = Config(
            distance=3,
            rounds=3,
            epochs=4,
            shots_per_epoch=100,
            calibration_shots=300,
            calibration_points=5,
        )
        calibration = build_calibration_curve(config)
        schedule_a = np.array([0.005, 0.005, 0.020, 0.020])
        schedule_b = np.array([0.005, 0.005, 0.010, 0.027])
        results_a, log_a = run_scenario(
            "step", 77, config, calibration, schedule_override=schedule_a
        )
        results_b, log_b = run_scenario(
            "step", 77, config, calibration, schedule_override=schedule_b
        )
        pd.testing.assert_frame_equal(
            results_a[results_a["epoch"] < 2].reset_index(drop=True),
            results_b[results_b["epoch"] < 2].reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            log_a[log_a["epoch"] < 2].reset_index(drop=True),
            log_b[log_b["epoch"] < 2].reset_index(drop=True),
        )


if __name__ == "__main__":
    unittest.main()
