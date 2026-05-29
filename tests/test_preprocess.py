from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocess import PreprocessConfig, build_preprocessed_dataset


DATA_DIR = PROJECT_ROOT / "A股数据"


class PreprocessTest(unittest.TestCase):
    def test_preprocess_dates_do_not_leak_future_features(self) -> None:
        config = PreprocessConfig(
            data_dir=str(DATA_DIR),
            start_date=20260105,
            end_date=20260109,
        )
        dataset, summary = build_preprocessed_dataset(config)

        self.assertGreater(len(dataset), 0)
        self.assertTrue((dataset["feature_end_date"] < dataset["trade_date"]).all())
        self.assertTrue((dataset["trade_date"] < dataset["label_end_date"]).all())
        self.assertLessEqual(summary.max_pool_size, 300)
        self.assertGreaterEqual(summary.min_pool_size, 250)

    def test_features_have_no_missing_values_after_cross_section_fill(self) -> None:
        config = PreprocessConfig(
            data_dir=str(DATA_DIR),
            start_date=20260105,
            end_date=20260106,
        )
        dataset, summary = build_preprocessed_dataset(config)
        feature_columns = summary.feature_columns

        self.assertFalse(dataset[feature_columns].isna().any().any())
        self.assertNotIn("close_t1", dataset.columns)
        self.assertNotIn("future_return", dataset.columns)


if __name__ == "__main__":
    unittest.main()
