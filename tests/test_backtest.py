from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtest import run_score_backtest
from stock_pool import build_stock_pool


DATA_DIR = PROJECT_ROOT / "A股数据"


class BacktestTest(unittest.TestCase):
    def test_backtest_outputs_curve_holdings_trades_and_metrics(self) -> None:
        rows = []
        for trade_date in [20260105, 20260106, 20260107]:
            pool, _ = build_stock_pool(DATA_DIR, as_of_date=trade_date)
            for rank, code in enumerate(pool["ts_code"].head(20)):
                rows.append(
                    {
                        "trade_date": trade_date,
                        "ts_code": code,
                        "score": float(20 - rank),
                    }
                )
        predictions = pd.DataFrame(rows)
        curve, holdings, trades, metrics = run_score_backtest(
            predictions,
            data_dir=DATA_DIR,
            n_holdings=10,
            rebalance_count=2,
            enable_risk_control=False,
        )

        self.assertEqual(len(curve), 3)
        self.assertFalse(holdings.empty)
        self.assertFalse(trades.empty)
        self.assertEqual(metrics.trading_days, 3)

    def test_buffered_strategy_keeps_holdings_inside_rank_buffer(self) -> None:
        pool, _ = build_stock_pool(DATA_DIR, as_of_date=20260105)
        codes = pool["ts_code"].head(40).tolist()
        rows = []
        for rank, code in enumerate(codes):
            rows.append({"trade_date": 20260105, "ts_code": code, "score": float(40 - rank)})
        second_day_order = codes[10:30] + codes[:10] + codes[30:40]
        for rank, code in enumerate(second_day_order):
            rows.append({"trade_date": 20260106, "ts_code": code, "score": float(40 - rank)})

        _, holdings, trades, _ = run_score_backtest(
            pd.DataFrame(rows),
            data_dir=DATA_DIR,
            n_holdings=10,
            strategy="buffered_topn",
            max_sell=2,
            buffer_rank=30,
            enable_risk_control=False,
        )

        self.assertEqual(set(trades["action"]), {"buy"})
        self.assertEqual(len(trades), 10)
        self.assertEqual(holdings.groupby("trade_date")["ts_code"].nunique().max(), 10)

    def test_buffered_strategy_limits_daily_sells(self) -> None:
        pool, _ = build_stock_pool(DATA_DIR, as_of_date=20260105)
        codes = pool["ts_code"].head(40).tolist()
        rows = []
        for rank, code in enumerate(codes):
            rows.append({"trade_date": 20260105, "ts_code": code, "score": float(40 - rank)})
        second_day_order = codes[10:40] + codes[:10]
        for rank, code in enumerate(second_day_order):
            rows.append({"trade_date": 20260106, "ts_code": code, "score": float(40 - rank)})

        _, _, trades, _ = run_score_backtest(
            pd.DataFrame(rows),
            data_dir=DATA_DIR,
            n_holdings=10,
            strategy="buffered_topn",
            max_sell=2,
            buffer_rank=30,
            enable_risk_control=False,
        )

        second_day_trades = trades[trades["trade_date"] == 20260106]
        self.assertEqual((second_day_trades["action"] == "sell").sum(), 2)
        self.assertEqual((second_day_trades["action"] == "buy").sum(), 2)

    def test_risk_control_limits_industry_concentration_on_buys(self) -> None:
        basic = pd.read_csv(DATA_DIR / "basic.csv", dtype={"ts_code": str})
        industry = basic["industry"].dropna().value_counts().index[0]
        same_industry_codes = basic[basic["industry"] == industry]["ts_code"].head(10).tolist()
        other_codes = basic[basic["industry"] != industry]["ts_code"].head(10).tolist()
        rows = []
        for rank, code in enumerate(same_industry_codes + other_codes):
            rows.append({"trade_date": 20260105, "ts_code": code, "score": float(30 - rank)})

        _, holdings, trades, _ = run_score_backtest(
            pd.DataFrame(rows),
            data_dir=DATA_DIR,
            n_holdings=10,
            strategy="buffered_topn",
            max_industry_count=3,
            max_daily_volatility=1.0,
        )

        first_day_holdings = holdings[holdings["trade_date"] == 20260105]
        self.assertLessEqual((first_day_holdings["industry"] == industry).sum(), 3)
        self.assertIn("industry", trades.columns)
        self.assertIn("risk_reason", trades.columns)

    def test_risk_control_can_sell_high_volatility_holdings(self) -> None:
        pool, _ = build_stock_pool(DATA_DIR, as_of_date=20260105)
        codes = pool["ts_code"].head(20).tolist()
        rows = []
        for trade_date in [20260105, 20260106]:
            for rank, code in enumerate(codes):
                rows.append({"trade_date": trade_date, "ts_code": code, "score": float(20 - rank)})

        _, _, trades, _ = run_score_backtest(
            pd.DataFrame(rows),
            data_dir=DATA_DIR,
            n_holdings=10,
            strategy="buffered_topn",
            max_sell=2,
            buffer_rank=30,
            max_daily_volatility=0.000001,
        )

        second_day_sells = trades[
            (trades["trade_date"] == 20260106) & (trades["action"] == "sell")
        ]
        self.assertGreater(len(second_day_sells), 0)
        self.assertTrue((second_day_sells["risk_reason"] == "high_volatility").any())


if __name__ == "__main__":
    unittest.main()
