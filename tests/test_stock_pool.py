from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_pool import build_stock_pool, read_latest_index_weights


DATA_DIR = PROJECT_ROOT / "A股数据"


class StockPoolTest(unittest.TestCase):
    def test_latest_may_2026_falls_back_to_non_empty_weight_snapshot(self) -> None:
        pool, summary = build_stock_pool(DATA_DIR, as_of_date=20260527)

        self.assertGreater(summary.final_count, 250)
        self.assertLessEqual(summary.final_count, 300)
        self.assertEqual(summary.weight_trade_date, 20260401)
        self.assertIn("202604_000300.SH.csv", summary.weight_file)
        self.assertEqual(pool["ts_code"].nunique(), len(pool))
        self.assertEqual(set(pool["trade_date"]), {20260527})
        self.assertEqual(set(pool["weight_trade_date"]), {20260401})

    def test_strict_weight_lookup_does_not_use_same_day_snapshot(self) -> None:
        _, snapshot = read_latest_index_weights(DATA_DIR, 20260401)

        self.assertEqual(snapshot.trade_date, 20260331)
        self.assertIn("202603_000300.SH.csv", str(snapshot.path))

    def test_weight_file_with_multiple_snapshots_keeps_only_latest_valid_snapshot(self) -> None:
        pool, summary = build_stock_pool(DATA_DIR, as_of_date=20250102)

        self.assertLessEqual(summary.index_member_count, 300)
        self.assertLessEqual(summary.final_count, 300)
        self.assertEqual(summary.weight_trade_date, 20241231)
        self.assertEqual(pool["weight_trade_date"].nunique(), 1)

    def test_pool_excludes_st_and_beijing_market(self) -> None:
        pool, _ = build_stock_pool(DATA_DIR, as_of_date=20260527)

        self.assertNotIn("北交所", set(pool["market"]))
        self.assertFalse(pool["name"].astype(str).str.contains("ST", regex=False).any())


if __name__ == "__main__":
    unittest.main()
