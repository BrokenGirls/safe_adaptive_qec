"""Fast structural checks for the online adaptation experiment."""

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_online_adaptation import Config, make_circuit, scenario_schedule  # noqa: E402


class SmokeTests(unittest.TestCase):
    def test_generated_circuit_has_detectors_and_observable(self) -> None:
        circuit = make_circuit(
            basis="x",
            distance=3,
            rounds=3,
            base_error_rate=0.005,
            measurement_error_rate=0.01,
        )
        self.assertGreater(circuit.num_detectors, 0)
        self.assertEqual(circuit.num_observables, 1)

    def test_step_schedule_changes_once(self) -> None:
        config = Config(epochs=9)
        schedule = scenario_schedule("step", config)
        self.assertEqual(len(schedule), 9)
        self.assertEqual(schedule[0], config.initial_measurement_error)
        self.assertEqual(schedule[-1], config.high_measurement_error)
        self.assertEqual(np.count_nonzero(np.diff(schedule)), 1)


if __name__ == "__main__":
    unittest.main()

