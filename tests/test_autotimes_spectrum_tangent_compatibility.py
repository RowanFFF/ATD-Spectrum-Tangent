import unittest

import numpy as np

from scripts.autotimes_spectrum_tangent_compatibility import (
    block_rows,
    fit_from_stats,
    select_and_summarize,
)


class AutoTimesSpectrumTangentCompatibilityTest(unittest.TestCase):
    def test_nonnegative_fit_falls_back_to_identity(self):
        gamma, gain, r2 = fit_from_stats(2.0, 1.0, -0.5)
        self.assertEqual(gamma, 0.0)
        self.assertEqual(gain, 0.0)
        self.assertEqual(r2, 0.0)

    def test_selection_pools_first_three_blocks_and_confirms_on_fourth(self):
        rows = []
        for seed in (2021, 2022, 2023):
            # [origin, period, V, A, B].  Period 24 has the stronger positive
            # direction on blocks 1--3 and remains positive on block 4.
            moments = np.zeros((16, 2, 3), dtype=np.float64)
            moments[:, :, 0] = 1.0
            moments[:, :, 1] = 1.0
            moments[:, 0, 2] = 0.1
            moments[:, 1, 2] = 0.3
            rows.extend(block_rows(
                width=2,
                seed=seed,
                moments=moments,
                periods=(12, 24),
            ))
        summary = select_and_summarize(
            rows, widths=(2,), seeds=(2021, 2022, 2023),
            periods=(12, 24),
        )[0]
        self.assertEqual(summary["selected_period"], 24)
        self.assertAlmostEqual(summary["fit_positive_gamma"], 0.3)
        self.assertEqual(summary["seed_period_agreement"], 3)
        self.assertEqual(summary["confirmation_seed_wins"], 3)


if __name__ == "__main__":
    unittest.main()
