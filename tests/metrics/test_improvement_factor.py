"""Tests for improvement factor metric."""
from unittest import TestCase

from blackwater.metrics.improvement_factor import improvement_factor, Problem, Trial


class TestImprovementFactor(TestCase):
    """TestImprovementFactor."""

    def test_improvement_factor_calculation(self):
        """Tests improvement factor metric calculation."""
        factor = improvement_factor(
            problems=[
                Problem(
                    trials=[Trial(noisy=1.0, mitigated=2.0)],
                    ideal_exp_value=0.0,
                    circuit=None,
                    observable=None,
                )
            ],
            n_shots=1,
            n_mitigation_shots=1,
        )

        self.assertEqual(0.5, factor)

    def test_with_number_of_parameters(self):
        """Test improvement factor with different parameters."""
        factor = improvement_factor(
            problems=[
                Problem(
                    trials=[
                        Trial(noisy=3.0, mitigated=4.0),
                        Trial(noisy=1.0, mitigated=2.0),
                    ],
                    ideal_exp_value=2.0,
                ),
                Problem(trials=[Trial(noisy=3.0, mitigated=4.0)], ideal_exp_value=2.0),
            ],
            n_shots=3,
            n_mitigation_shots=2,
        )

        self.assertEqual(0.75, factor)

    def test_alternative_arguments(self):
        """Tests alternative function arguments."""
        factor1 = improvement_factor([(0.0, [(1.0, 2.0)])], 1, 1)
        factor2 = improvement_factor(
            [(2.0, [(3.0, 4.0), (1.0, 2.0)]), (2.0, [(3.0, 4.0)])], 3, 2
        )

        self.assertEqual(0.5, factor1)
        self.assertEqual(0.75, factor2)
