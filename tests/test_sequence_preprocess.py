from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocess_seq import SequencePreprocessConfig, build_sequence_dataset


DATA_DIR = PROJECT_ROOT / "A股数据"


class SequencePreprocessTest(unittest.TestCase):
    def test_sequence_dataset_has_fixed_window_and_no_leakage(self) -> None:
        config = SequencePreprocessConfig(
            data_dir=str(DATA_DIR),
            start_date=20260105,
            end_date=20260106,
        )
        splits, summary = build_sequence_dataset(config)
        dataset = splits["test"]

        self.assertGreater(dataset["X"].shape[0], 0)
        self.assertEqual(dataset["X"].ndim, 3)
        self.assertEqual(dataset["X"].shape[1], 20)
        self.assertEqual(dataset["X"].shape[2], summary.feature_count)
        self.assertTrue(np.isfinite(dataset["X"]).all())
        self.assertTrue(np.isfinite(dataset["y"]).all())
        self.assertTrue((dataset["feature_dates"].max(axis=1) < dataset["trade_date"]).all())
        self.assertTrue((dataset["trade_date"] < dataset["label_end_date"]).all())


if __name__ == "__main__":
    unittest.main()

