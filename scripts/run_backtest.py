from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtest import json_safe, run_score_backtest, write_backtest_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest score-ranked stock predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data-dir", default="A股数据")
    parser.add_argument("--output-dir", default="outputs/backtest")
    parser.add_argument("--index-code", default="000300.SH")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument(
        "--strategy",
        choices=["simple_topk", "buffered_topn"],
        default="buffered_topn",
        help="Trading strategy. buffered_topn keeps holdings inside the rank buffer.",
    )
    parser.add_argument("--n-holdings", type=int, default=10)
    parser.add_argument("--max-sell", type=int, default=2)
    parser.add_argument("--buffer-rank", type=int, default=30)
    parser.add_argument(
        "--disable-risk-control",
        action="store_true",
        help="Disable volatility and industry concentration controls.",
    )
    parser.add_argument("--max-industry-count", type=int, default=3)
    parser.add_argument("--volatility-window", type=int, default=20)
    parser.add_argument("--max-daily-volatility", type=float, default=0.08)
    parser.add_argument(
        "--rebalance-count",
        type=int,
        default=None,
        help="Backward-compatible alias for --max-sell.",
    )
    parser.add_argument("--commission-rate", type=float, default=0.00025, help="Commission rate (default 0.025%%)")
    parser.add_argument("--stamp-tax-rate", type=float, default=0.001, help="Stamp tax rate on sells (default 0.1%%)")
    parser.add_argument("--weight-method", type=str, choices=["equal", "gmv"], default="equal", help="Weight allocation method.")
    parser.add_argument("--gmv-volatility-target", type=float, default=0.12, help="Target annualized volatility for GMV.")
    parser.add_argument("--gmv-penalty-lambda", type=float, default=0.01, help="Turnover penalty lambda for GMV.")
    parser.add_argument("--gmv-cov-window", type=int, default=60, help="Lookback window for GMV covariance.")
    parser.add_argument("--gmv-min-weight", type=float, default=0.05, help="Minimum weight for single stock in GMV.")
    parser.add_argument("--gmv-max-weight", type=float, default=0.35, help="Maximum weight for single stock in GMV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions = pd.read_csv(args.predictions, dtype={"ts_code": str})
    curve, holdings, trades, metrics = run_score_backtest(
        predictions=predictions,
        data_dir=args.data_dir,
        index_code=args.index_code,
        initial_cash=args.initial_cash,
        n_holdings=args.n_holdings,
        strategy=args.strategy,
        max_sell=args.max_sell,
        buffer_rank=args.buffer_rank,
        rebalance_count=args.rebalance_count,
        enable_risk_control=not args.disable_risk_control,
        max_industry_count=args.max_industry_count,
        volatility_window=args.volatility_window,
        max_daily_volatility=args.max_daily_volatility,
        commission_rate=args.commission_rate,
        stamp_tax_rate=args.stamp_tax_rate,
        weight_method=args.weight_method,
        gmv_volatility_target=args.gmv_volatility_target,
        gmv_penalty_lambda=args.gmv_penalty_lambda,
        gmv_cov_window=args.gmv_cov_window,
        gmv_min_weight=args.gmv_min_weight,
        gmv_max_weight=args.gmv_max_weight,
    )
    write_backtest_outputs(args.output_dir, curve, holdings, trades, metrics)
    print(json.dumps(json_safe(asdict(metrics)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
