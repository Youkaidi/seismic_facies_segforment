import unittest

import numpy as np

from evaluation.geological_metrics import (
    evaluate_geological_consistency,
    transition_count_matrix,
    transition_probability_matrix,
)


class GeologicalMetricsTest(unittest.TestCase):
    def setUp(self):
        self.target = np.array([[0, 0, 1, 1, 2, 2]], dtype=np.int64)
        self.target_counts = transition_count_matrix(self.target, 3, vertical_axis=1)

    def test_transition_counts_and_probabilities(self):
        expected = np.array(
            [[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.int64
        )
        np.testing.assert_array_equal(self.target_counts, expected)
        probabilities = transition_probability_matrix(self.target_counts)
        np.testing.assert_allclose(
            probabilities,
            np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]),
        )

    def test_reverse_rate_and_vtpm_distance(self):
        prediction = np.array([[0, 0, 2, 1, 2, 2]], dtype=np.int64)
        metrics, matrices = evaluate_geological_consistency(
            prediction,
            self.target,
            num_classes=3,
            vertical_axis=1,
        )
        self.assertAlmostEqual(metrics["reverse_transition_rate"], 1.0 / 5.0)
        self.assertGreater(metrics["vtpm_frobenius_to_target"], 0.0)
        self.assertIn("prediction_vtpm", matrices)
        self.assertIn("target_vtpm", matrices)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_geological_consistency(
                np.zeros((2, 3), dtype=np.int64),
                np.zeros((3, 2), dtype=np.int64),
                num_classes=3,
            )


if __name__ == "__main__":
    unittest.main()
