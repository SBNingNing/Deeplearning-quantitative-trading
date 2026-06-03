"""Generate buy/sell recommendations for the next trading day.

Usage (after daily data sync):
    python scripts/predict_latest.py
    python scripts/predict_latest.py --weight-method gmv

This script:
1. Finds the latest available trade date
2. Builds the stock pool and generates model predictions
3. Reads current holdings (if any)
4. Determines buy/sell using the buffered top-N strategy
5. If --weight-method gmv, optimizes portfolio weights with score tilt
6. Outputs actionable trading recommendations
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_pool import DEFAULT_INDEX_CODE, DEFAULT_MIN_AMOUNT, build_stock_pool
from preprocess import (
    available_daily_dates,
    finalized_feature_columns,
)
from preprocess_seq import build_sequence_features_for_date
from modeling import TemporalSegmentGRU, TemporalSegmentNet
from backtest import (
    apply_risk_annotations,
    prepare_daily_scores,
    rebalance_holdings,
    select_buy_candidates,
    read_industry_map,
)
from portfolio import (
    optimize_gmv_with_penalty,
    apply_volatility_control,
    compute_covariance_cache,
    portfolio_annualized_volatility,
)


def parse_yyyymmdd(value: int | str) -> datetime.date:
    """Parse an integer/string YYYYMMDD date."""
    return datetime.strptime(str(int(value)), "%Y%m%d").date()


def to_yyyymmdd(value) -> int:
    """Format a date object as integer YYYYMMDD."""
    return int(value.strftime("%Y%m%d"))


def next_business_day(value: int | str) -> int:
    """Weekend-only next business day helper for execution-date display."""
    day = parse_yyyymmdd(value) + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return to_yyyymmdd(day)


def iter_business_days(start: int, end: int) -> list[int]:
    """List weekend-only business days between start and end inclusive."""
    day = parse_yyyymmdd(start)
    end_day = parse_yyyymmdd(end)
    out: list[int] = []
    while day <= end_day:
        if day.weekday() < 5:
            out.append(to_yyyymmdd(day))
        day += timedelta(days=1)
    return out


def signal_for_execution_date(trading_dates: list[int], execution_date: int) -> int | None:
    """Return the immediately usable signal date for an execution date.

    For example, execution 20260601 uses signal 20260529. If the latest
    available signal does not map to the requested execution day, return None
    so the caller does not accidentally reuse stale data.
    """
    candidates = [date for date in trading_dates if date < execution_date]
    if not candidates:
        return None
    signal_date = candidates[-1]
    return signal_date if next_business_day(signal_date) == execution_date else None


def print_competition_schedule(trading_dates: list[int], start: int, end: int) -> None:
    """Print which competition days can be generated with current local data."""
    print("\nCompetition schedule readiness")
    print(f"{'Execution':<12s} {'Signal':<12s} {'Status'}")
    print("-" * 42)
    for execution_date in iter_business_days(start, end):
        signal_date = signal_for_execution_date(trading_dates, execution_date)
        if signal_date is None:
            previous = [date for date in trading_dates if date < execution_date]
            previous_text = str(previous[-1]) if previous else "-"
            print(f"{execution_date:<12d} {previous_text:<12s} WAIT_DATA")
        else:
            print(f"{execution_date:<12d} {signal_date:<12d} READY")


def read_close_prices(data_dir: Path, trade_date: int, codes: set[str]) -> dict[str, float]:
    """Read latest close prices for order sizing."""
    if not codes:
        return {}
    path = data_dir / "daily" / f"{trade_date}.csv"
    prices = pd.read_csv(
        path,
        usecols=["ts_code", "close"],
        dtype={"ts_code": str},
    )
    prices = prices[prices["ts_code"].isin(codes)].copy()
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    return prices.dropna(subset=["close"]).set_index("ts_code")["close"].to_dict()


def round_to_lot(shares: float, lot_size: int) -> int:
    """Round target shares down to a valid board lot."""
    if lot_size <= 0:
        return max(0, int(shares))
    return max(0, int(shares // lot_size) * lot_size)


def infer_current_shares(
    holdings_df: pd.DataFrame | None,
    prices: dict[str, float],
    portfolio_value: float,
    lot_size: int,
) -> dict[str, int]:
    """Read shares from holdings, or infer them from old weight files."""
    if holdings_df is None or holdings_df.empty:
        return {}

    out: dict[str, int] = {}
    if "shares" in holdings_df.columns:
        for _, row in holdings_df.iterrows():
            shares = pd.to_numeric(row.get("shares"), errors="coerce")
            if pd.notna(shares):
                out[str(row["ts_code"])] = max(0, int(shares))
        return out

    if "weight" in holdings_df.columns:
        for _, row in holdings_df.iterrows():
            code = str(row["ts_code"])
            price = prices.get(code)
            weight = pd.to_numeric(row.get("weight"), errors="coerce")
            if price and pd.notna(weight):
                out[code] = round_to_lot(portfolio_value * float(weight) / price, lot_size)
    return out


def infer_prev_weights_from_holdings(
    holdings_df: pd.DataFrame | None,
    prices: dict[str, float],
    portfolio_value: float,
) -> dict[str, float]:
    """Infer current portfolio weights from actual shares or value columns."""
    if holdings_df is None or holdings_df.empty or portfolio_value <= 0:
        return {}
    if "weight" in holdings_df.columns:
        return {
            str(row["ts_code"]): float(pd.to_numeric(row["weight"], errors="coerce"))
            for _, row in holdings_df.iterrows()
            if pd.notna(pd.to_numeric(row.get("weight"), errors="coerce"))
        }

    out: dict[str, float] = {}
    for _, row in holdings_df.iterrows():
        code = str(row["ts_code"])
        market_value = pd.to_numeric(row.get("market_value"), errors="coerce")
        if pd.isna(market_value):
            shares = pd.to_numeric(row.get("shares"), errors="coerce")
            price = pd.to_numeric(row.get("last_price"), errors="coerce")
            if pd.isna(price):
                price = pd.to_numeric(row.get("fill_price"), errors="coerce")
            if pd.isna(price):
                price = prices.get(code, np.nan)
            if pd.notna(shares) and pd.notna(price):
                market_value = float(shares) * float(price)
        if pd.notna(market_value):
            out[code] = float(market_value) / float(portfolio_value)
    return out


def build_order_plan(
    new_holdings: list[str],
    current_shares: dict[str, int],
    daily_weights: dict[str, float],
    prices: dict[str, float],
    day_scores: pd.DataFrame,
    portfolio_value: float,
    lot_size: int,
    buy_price_buffer: float,
    sell_price_buffer: float,
) -> pd.DataFrame:
    """Build actionable buy/sell/hold rows from target weights."""
    score_by_code = day_scores.set_index("ts_code", drop=False)
    ordered_codes = list(dict.fromkeys([*new_holdings, *current_shares.keys()]))
    rows: list[dict[str, object]] = []

    for code in ordered_codes:
        price = prices.get(code, np.nan)
        buy_price = price * (1.0 + buy_price_buffer) if pd.notna(price) else np.nan
        sell_price = price * (1.0 - sell_price_buffer) if pd.notna(price) else np.nan
        target_weight = float(daily_weights.get(code, 0.0))
        target_amount = portfolio_value * target_weight
        target_shares = (
            round_to_lot(target_amount / buy_price, lot_size)
            if pd.notna(buy_price) and buy_price > 0 and target_weight > 0
            else 0
        )
        current = int(current_shares.get(code, 0))
        delta = int(target_shares - current)
        if target_weight == 0.0 and current > 0:
            # Full liquidation should sell the actual position, including odd lots.
            delta = -current
            target_shares = 0
        action = "buy" if delta > 0 else ("sell" if delta < 0 else "hold")
        score_row = score_by_code.loc[code] if code in score_by_code.index else None

        rows.append(
            {
                "action": action,
                "ts_code": code,
                "current_shares": current,
                "target_shares": int(target_shares),
                "delta_shares": delta,
                "close_price": price,
                "est_price": buy_price if delta >= 0 else sell_price,
                "buy_price_buffer": buy_price_buffer,
                "sell_price_buffer": sell_price_buffer,
                "est_trade_value": (
                    abs(delta) * (buy_price if delta >= 0 else sell_price)
                    if pd.notna(price)
                    else np.nan
                ),
                "target_weight": target_weight,
                "target_amount": target_amount,
                "score": float(score_row["score"]) if score_row is not None else np.nan,
                "rank": int(score_row["rank"]) if score_row is not None else np.nan,
                "industry": str(score_row["industry"]) if score_row is not None else "",
            }
        )
    plan = pd.DataFrame(rows)
    if not plan.empty:
        plan = allocate_leftover_cash(plan, portfolio_value, lot_size)
    return plan


def allocate_leftover_cash(plan: pd.DataFrame, portfolio_value: float, lot_size: int) -> pd.DataFrame:
    """Use leftover cash to add valid lots to target holdings when possible."""
    out = plan.copy()
    if lot_size <= 0:
        return out
    while True:
        buy_price = out["close_price"] * (1.0 + out["buy_price_buffer"])
        invested = float((out["target_shares"] * buy_price).fillna(0.0).sum())
        cash = portfolio_value - invested
        candidates = out[
            (out["target_weight"] > 0)
            & buy_price.notna()
            & (buy_price > 0)
            & (buy_price * lot_size <= cash)
        ].copy()
        if candidates.empty:
            break
        candidates["underweight"] = candidates["target_amount"] - (
            candidates["target_shares"] * (candidates["close_price"] * (1.0 + candidates["buy_price_buffer"]))
        )
        idx = candidates.sort_values(["underweight", "target_weight"], ascending=False).index[0]
        out.loc[idx, "target_shares"] = int(out.loc[idx, "target_shares"]) + lot_size

    out["delta_shares"] = out["target_shares"].astype(int) - out["current_shares"].astype(int)
    sell_mask = (out["target_weight"] == 0.0) & (out["current_shares"] > 0)
    out.loc[sell_mask, "delta_shares"] = -out.loc[sell_mask, "current_shares"].astype(int)
    out["action"] = np.where(out["delta_shares"] > 0, "buy",
                             np.where(out["delta_shares"] < 0, "sell", "hold"))
    est_prices = np.where(
        out["delta_shares"] >= 0,
        out["close_price"] * (1.0 + out["buy_price_buffer"]),
        out["close_price"] * (1.0 - out["sell_price_buffer"]),
    )
    out["est_price"] = est_prices
    out["est_trade_value"] = out["delta_shares"].abs() * out["est_price"]
    return out


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint.get("model_type", "tsn")
    input_dim = checkpoint["input_dim"]
    hidden_dim = checkpoint.get("hidden_dim", 64)
    num_segments = checkpoint.get("num_segments", 4)
    dropout = checkpoint.get("dropout", 0.2)

    if model_type == "tsn_gru":
        model = TemporalSegmentGRU(input_dim, hidden_dim=hidden_dim,
                                   num_segments=num_segments, dropout=dropout)
    else:
        model = TemporalSegmentNet(input_dim, hidden_dim=hidden_dim,
                                   num_segments=num_segments, dropout=dropout)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily buy/sell recommendations.")
    parser.add_argument("--data-dir", default="A股数据")
    parser.add_argument("--trade-date", type=int, default=None,
                        help="Signal date to generate recommendations for. Defaults to latest daily data.")
    parser.add_argument("--execution-date", type=int, default=None,
                        help="Competition day to trade. The script uses the previous available signal date.")
    parser.add_argument("--schedule-start", type=int, default=None,
                        help="Print competition schedule readiness from this execution date.")
    parser.add_argument("--schedule-end", type=int, default=None,
                        help="Print competition schedule readiness to this execution date.")
    parser.add_argument("--print-schedule", action="store_true",
                        help="Only print schedule readiness and exit.")
    parser.add_argument("--checkpoint", default="outputs/models/tsn_full_best.pt")
    parser.add_argument("--holdings-csv", default="outputs/competition_holdings.csv")
    parser.add_argument("--actual-holdings-csv", default=None,
                        help="Optional actual filled holdings file. Used as current holdings source.")
    parser.add_argument("--update-holdings", action="store_true",
                        help="Overwrite holdings-csv with target holdings. Off by default for competition suggestions.")
    parser.add_argument("--reset-holdings", action="store_true",
                        help="Ignore existing holdings file and generate as an empty initial portfolio.")
    parser.add_argument("--portfolio-value", type=float, default=1_000_000.0,
                        help="Current portfolio value for amount calculation (default 100w).")
    parser.add_argument("--lot-size", type=int, default=100,
                        help="A-share trading lot size used for share rounding.")
    parser.add_argument("--buy-price-buffer", type=float, default=0.01,
                        help="Buy sizing price buffer over latest close, e.g. 0.01 means close*1.01.")
    parser.add_argument("--sell-price-buffer", type=float, default=0.005,
                        help="Sell value price buffer below latest close, e.g. 0.005 means close*0.995.")
    parser.add_argument("--orders-dir", default="outputs/orders",
                        help="Directory for actionable daily order files.")
    parser.add_argument("--n-holdings", type=int, default=10)
    parser.add_argument("--max-sell", type=int, default=2)
    parser.add_argument("--buffer-rank", type=int, default=30)
    parser.add_argument("--max-industry-count", type=int, default=3)
    parser.add_argument("--max-daily-volatility", type=float, default=0.08)
    parser.add_argument("--disable-risk-control", action="store_true")
    parser.add_argument("--weight-method", choices=["equal", "gmv"], default="gmv",
                        help="Weight allocation: equal or gmv (score-tilted GMV).")
    parser.add_argument("--gmv-penalty-lambda", type=float, default=0.005)
    parser.add_argument("--gmv-score-tilt", type=float, default=0.003,
                        help="Score tilt strength in GMV objective. Set 0 for pure GMV.")
    parser.add_argument("--gmv-cov-window", type=int, default=60)
    parser.add_argument("--gmv-min-weight", type=float, default=0.05)
    parser.add_argument("--gmv-max-weight", type=float, default=0.20)
    parser.add_argument("--gmv-volatility-target", type=float, default=0.12)
    parser.add_argument("--allow-cash-vol-control", action="store_true",
                        help="Allow target-volatility control to scale down stock weights into cash.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    device = torch.device(args.device)
    enable_risk_control = not args.disable_risk_control

    # 1. Find latest trade date ------------------------------------------------------------------
    trading_dates = available_daily_dates(data_dir)
    if not trading_dates:
        print("ERROR: No daily data found. Please sync data from cloud drive first.")
        return 1

    date_to_index = {date: idx for idx, date in enumerate(trading_dates)}
    if args.print_schedule or args.schedule_start or args.schedule_end:
        if args.schedule_start is None or args.schedule_end is None:
            print("ERROR: --schedule-start and --schedule-end are required for schedule mode.")
            return 1
        print_competition_schedule(trading_dates, args.schedule_start, args.schedule_end)
        if args.print_schedule:
            return 0

    if args.trade_date is not None and args.execution_date is not None:
        print("ERROR: Use either --trade-date or --execution-date, not both.")
        return 1

    execution_date = None
    if args.execution_date is not None:
        latest_date = signal_for_execution_date(trading_dates, args.execution_date)
        if latest_date is None:
            previous = [date for date in trading_dates if date < args.execution_date]
            previous_text = previous[-1] if previous else "none"
            print(f"ERROR: Cannot generate execution-date {args.execution_date} yet.")
            print(f"Previous available signal date: {previous_text}")
            print("Reason: each competition day needs the previous trading day's close data.")
            print("Sync the latest daily data after market close, then run again.")
            return 1
        execution_date = args.execution_date
    else:
        latest_date = args.trade_date or trading_dates[-1]
        execution_date = next_business_day(latest_date)

    if latest_date not in date_to_index:
        print(f"ERROR: trade date {latest_date} not found in daily data.")
        print(f"Latest available date: {trading_dates[-1]}")
        return 1
    idx = date_to_index[latest_date]
    if idx < args.window:
        print(f"ERROR: Not enough history before {latest_date} (need {args.window} days).")
        return 1

    feature_end_date = trading_dates[idx - 1]
    window_dates = trading_dates[idx - args.window : idx]

    print(f"Signal date       : {latest_date}")
    print(f"Execution date    : {execution_date}")
    print(f"Feature end date  : {feature_end_date}")
    print(f"Weight method     : {args.weight_method}")
    print()

    # 2. Stock pool ----------------------------------------------------------------------------
    pool, summary = build_stock_pool(
        data_dir, as_of_date=feature_end_date,
        index_code=DEFAULT_INDEX_CODE, min_amount=DEFAULT_MIN_AMOUNT,
    )
    print(f"Stock pool: {summary.final_count} stocks")

    # 3. Predictions ---------------------------------------------------------------------------
    print("Generating predictions...")
    seq_frame = build_sequence_features_for_date(data_dir, pool, window_dates)
    if seq_frame.empty:
        print("ERROR: No sequence features generated.")
        return 1

    feature_columns = finalized_feature_columns(seq_frame)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        return 1

    model, ckpt = load_model(str(checkpoint_path), device)
    ckpt_features = ckpt.get("feature_columns", feature_columns)
    usable_features = [c for c in ckpt_features if c in seq_frame.columns]
    if not usable_features:
        print("ERROR: No matching feature columns.")
        return 1

    scores = []
    for ts_code, group in seq_frame.groupby("ts_code", sort=False):
        if group.shape[0] != args.window:
            continue
        group_sorted = group.sort_values("trade_date")
        x = torch.from_numpy(
            group_sorted[usable_features].to_numpy(dtype=np.float32)
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            scores.append({"ts_code": ts_code, "score": float(model(x).cpu().item())})

    if not scores:
        print("ERROR: No predictions generated.")
        return 1

    pred_df = pd.DataFrame(scores).sort_values("score", ascending=False)
    pred_df["trade_date"] = latest_date

    # 4. Current holdings ----------------------------------------------------------------------
    holdings_path = Path(args.holdings_csv)
    holdings_source_path = Path(args.actual_holdings_csv) if args.actual_holdings_csv else holdings_path
    current_holdings: list[str] = []
    prev_weights: dict[str, float] = {}
    holdings_df: pd.DataFrame | None = None
    if args.reset_holdings:
        print("Reset holdings enabled - treating as initial empty portfolio.")
        holdings_path.parent.mkdir(parents=True, exist_ok=True)
    elif holdings_source_path.exists():
        holdings_df = pd.read_csv(holdings_source_path, dtype={"ts_code": str})
        current_holdings = holdings_df["ts_code"].tolist()
        if "weight" in holdings_df.columns:
            prev_weights = dict(zip(holdings_df["ts_code"], holdings_df["weight"]))
        print(f"Holdings source: {holdings_source_path}")
        print(f"Current holdings ({len(current_holdings)}): {current_holdings}")
    else:
        print(f"No holdings file found at {holdings_source_path} - initial day.")
        holdings_path.parent.mkdir(parents=True, exist_ok=True)

    # 5. Risk annotations ----------------------------------------------------------------------
    day_scores = prepare_daily_scores(pred_df)
    industry_map = read_industry_map(data_dir)
    day_scores = apply_risk_annotations(
        day_scores=day_scores, data_dir=data_dir, trade_date=latest_date,
        trading_dates=trading_dates, date_to_index=date_to_index,
        industry_map=industry_map, enable_risk_control=enable_risk_control,
        volatility_window=20, max_daily_volatility=args.max_daily_volatility,
    )

    # 6. Determine buys/sells ------------------------------------------------------------------
    if not current_holdings:
        buys = select_buy_candidates(
            day_scores=day_scores, existing_holdings=[], excluded_codes=set(),
            slots=args.n_holdings, max_industry_count=args.max_industry_count,
            enable_risk_control=enable_risk_control,
        )
        sells = []
        new_holdings = buys
    else:
        new_holdings, sells, buys = rebalance_holdings(
            holdings=current_holdings, day_scores=day_scores,
            n_holdings=args.n_holdings, max_sell=args.max_sell,
            buffer_rank=args.buffer_rank, strategy="buffered_topn",
            max_industry_count=args.max_industry_count,
            enable_risk_control=enable_risk_control,
        )

    if holdings_df is not None and current_holdings:
        current_price_map = read_close_prices(data_dir, latest_date, set(current_holdings))
        inferred_weights = infer_prev_weights_from_holdings(
            holdings_df,
            prices=current_price_map,
            portfolio_value=args.portfolio_value,
        )
        if inferred_weights:
            prev_weights = inferred_weights

    # 7. Weight optimization (GMV) --------------------------------------------------------------
    daily_weights: dict[str, float] = {}
    cash_weight = 0.0
    cov_matrix = None
    pre_control_vol = 0.0
    post_control_vol = 0.0

    if args.weight_method == "gmv" and new_holdings:
        print("Running GMV optimization...")
        cov_matrix = compute_covariance_cache(
            data_dir=data_dir, trading_dates=trading_dates,
            date_to_index=date_to_index, trade_date=latest_date,
            codes=new_holdings, window=args.gmv_cov_window,
        )
        prev_w = pd.Series(prev_weights, dtype=float).reindex(new_holdings).fillna(0.0)
        selected_scores = day_scores.set_index("ts_code")["score"].reindex(new_holdings)
        score_std = selected_scores.std(ddof=0)
        if pd.notna(score_std) and score_std > 0:
            score_alpha = (selected_scores - selected_scores.mean()) / score_std
        else:
            score_alpha = pd.Series(0.0, index=new_holdings)
        opt_w = optimize_gmv_with_penalty(
            cov_matrix, prev_weights=prev_w,
            score_alpha=score_alpha,
            penalty_lambda=args.gmv_penalty_lambda,
            score_tilt=args.gmv_score_tilt,
            min_weight=args.gmv_min_weight,
            max_weight=args.gmv_max_weight,
        )
        pre_control_vol = portfolio_annualized_volatility(opt_w, cov_matrix)
        if args.allow_cash_vol_control:
            final_w, cash_weight = apply_volatility_control(
                opt_w, cov_matrix, target_vol=args.gmv_volatility_target,
            )
        else:
            final_w = opt_w
            cash_weight = 0.0
        post_control_vol = portfolio_annualized_volatility(final_w, cov_matrix)
        daily_weights = final_w.to_dict()
        print(
            f"  GMV diagnostics: pre_vol={pre_control_vol*100:.2f}% "
            f"post_vol={post_control_vol*100:.2f}% "
            f"min_w={final_w.min()*100:.2f}% max_w={final_w.max()*100:.2f}% "
            f"score_tilt={args.gmv_score_tilt:g}"
        )
        if args.allow_cash_vol_control and cash_weight > 0.001:
            print(f"  Volatility trigger: scaled to {cash_weight*100:.1f}% cash")
        elif not args.allow_cash_vol_control:
            print("  Full-investment mode: target volatility is reported but not converted to cash")
    else:
        eq = 1.0 / max(len(new_holdings), 1)
        daily_weights = {c: eq for c in new_holdings}

    # 8. Build executable order plan ------------------------------------------------------------
    order_codes = set(new_holdings) | set(current_holdings)
    close_prices = read_close_prices(data_dir, latest_date, order_codes)
    current_shares = infer_current_shares(
        holdings_df,
        prices=close_prices,
        portfolio_value=args.portfolio_value,
        lot_size=args.lot_size,
    )
    order_plan = build_order_plan(
        new_holdings=new_holdings,
        current_shares=current_shares,
        daily_weights=daily_weights,
        prices=close_prices,
        day_scores=day_scores,
        portfolio_value=args.portfolio_value,
        lot_size=args.lot_size,
        buy_price_buffer=args.buy_price_buffer,
        sell_price_buffer=args.sell_price_buffer,
    )
    estimated_buy_value = float(
        order_plan.loc[order_plan["delta_shares"] > 0, "est_trade_value"].sum()
    )
    estimated_sell_value = float(
        order_plan.loc[order_plan["delta_shares"] < 0, "est_trade_value"].sum()
    )
    estimated_position_value = float(
        (order_plan["target_shares"] * order_plan["est_price"]).fillna(0.0).sum()
    )
    estimated_cash = args.portfolio_value - estimated_position_value

    # 9. Print recommendations ------------------------------------------------------------------
    border = "=" * 80
    print(f"\n{border}")
    print("  TRADING RECOMMENDATIONS")
    print(border)

    if sells:
        print(f"\n  >>> SELL ({len(sells)} stocks):")
        for code in sells:
            r = day_scores[day_scores["ts_code"] == code]
            info = f"  score={r.iloc[0]['score']:.6f}  rank={int(r.iloc[0]['rank'])}" if not r.empty else ""
            print(f"    {code}{info}")
    else:
        print("\n  >>> SELL: None")

    if buys:
        print(f"\n  >>> BUY ({len(buys)} stocks):")
        for code in buys:
            r = day_scores[day_scores["ts_code"] == code]
            info = f"  score={r.iloc[0]['score']:.6f}  rank={int(r.iloc[0]['rank'])}" if not r.empty else ""
            print(f"    {code}{info}")
    else:
        print("\n  >>> BUY: None (no changes needed)")

    executable_orders = order_plan[order_plan["delta_shares"] != 0].copy()
    print(
        f"\n  ORDER LIST  (execute={execution_date}, "
        f"estimated by close={latest_date}, lot={args.lot_size})"
    )
    if executable_orders.empty:
        print("    No executable share changes.")
    else:
        print(
            f"  {'Action':<6s} {'Stock':<12s} {'Shares':>9s} {'Price':>9s} "
            f"{'Est.Value':>12s} {'Target':>8s} {'Rank':>5s}"
        )
        print(f"  {'-' * 68}")
        for _, row in executable_orders.iterrows():
            action = str(row["action"]).upper()
            shares = abs(int(row["delta_shares"]))
            print(
                f"  {action:<6s} {row['ts_code']:<12s} {shares:>9,d} "
                f"{row['est_price']:>9.2f} {row['est_trade_value']:>12,.0f} "
                f"{row['target_weight']*100:>7.2f}% {int(row['rank']):>5d}"
            )
        print(
            f"  Estimated buy={estimated_buy_value:,.0f}, "
            f"sell={estimated_sell_value:,.0f}, cash after rounding={estimated_cash:,.0f}"
        )

    # Weight table
    print(f"\n  TARGET PORTFOLIO  (method={args.weight_method}, "
          f"value={args.portfolio_value:,.0f})")
    print(f"  {'#':<3s} {'Stock':<12s} {'Weight':>8s} {'Amount':>12s} "
          f"{'Shares':>9s} {'Price':>8s} {'Score':>10s} {'Rank':>5s} {'Industry':<12s}")
    print(f"  {'-' * 96}")

    order_by_code = order_plan.set_index("ts_code", drop=False)
    for i, code in enumerate(new_holdings, 1):
        w = daily_weights.get(code, 0)
        amount = args.portfolio_value * w
        r = day_scores[day_scores["ts_code"] == code]
        target_shares = int(order_by_code.at[code, "target_shares"]) if code in order_by_code.index else 0
        est_price = float(order_by_code.at[code, "est_price"]) if code in order_by_code.index else np.nan
        close_price = float(order_by_code.at[code, "close_price"]) if code in order_by_code.index else np.nan
        if not r.empty:
            row = r.iloc[0]
            print(f"  {i:<3d} {code:<12s} {w*100:>7.2f}% {amount:>11,.0f} "
                  f"{target_shares:>9,d} {close_price:>8.2f} "
                  f"{row['score']:>10.6f} {int(row['rank']):>5d} {row['industry']:<12s}")
        else:
            print(f"  {i:<3d} {code:<12s} {w*100:>7.2f}% {amount:>11,.0f} "
                  f"{target_shares:>9,d} {close_price:>8.2f}")

    if cash_weight > 0.001:
        cash_amount = args.portfolio_value * cash_weight
        print(f"  {'':>3s} {'(cash)':<12s} {cash_weight*100:>7.2f}% {cash_amount:>11,.0f}")

    print(f"\n{border}")

    # 10. Save target reference and orders ------------------------------------------------------
    holdings_rows = []
    for code in new_holdings:
        row = order_by_code.loc[code]
        holdings_rows.append(
            {
                "ts_code": code,
                "buy_date": execution_date,
                "weight": daily_weights.get(code, 0.0),
                "shares": int(row["target_shares"]),
                "last_price": float(row["close_price"]),
                "sizing_price": float(row["est_price"]),
                "market_value": int(row["target_shares"]) * float(row["close_price"]),
                "sizing_value": int(row["target_shares"]) * float(row["est_price"]),
            }
        )
    holdings_out = pd.DataFrame(holdings_rows)

    orders_dir = Path(args.orders_dir)
    orders_dir.mkdir(parents=True, exist_ok=True)
    target_holdings_path = orders_dir / f"target_holdings_{execution_date}.csv"
    holdings_out.to_csv(target_holdings_path, index=False)
    print(f"Target holdings reference saved: {target_holdings_path}")
    if args.update_holdings:
        holdings_out.to_csv(holdings_path, index=False)
        print(f"Actual holdings overwritten: {holdings_path}")
    else:
        print(f"Actual holdings NOT changed: {holdings_path}")

    orders_path = orders_dir / f"orders_{execution_date}.csv"
    order_plan.to_csv(orders_path, index=False)
    print(f"Orders saved: {orders_path}")

    state_path = orders_dir / f"state_{execution_date}.json"
    state = {
        "trade_date": int(latest_date),
        "signal_date": int(latest_date),
        "execution_date": int(execution_date),
        "portfolio_value": float(args.portfolio_value),
        "estimated_position_value": estimated_position_value,
        "estimated_cash_after_rounding": estimated_cash,
        "estimated_buy_value": estimated_buy_value,
        "estimated_sell_value": estimated_sell_value,
        "lot_size": int(args.lot_size),
        "price_basis": "latest_close_with_execution_buffer",
        "holdings_source": str(holdings_source_path),
        "target_holdings_path": str(target_holdings_path),
        "actual_holdings_updated": bool(args.update_holdings),
        "buy_price_buffer": float(args.buy_price_buffer),
        "sell_price_buffer": float(args.sell_price_buffer),
        "weight_method": args.weight_method,
        "cash_weight": float(cash_weight),
        "pre_control_annualized_vol": float(pre_control_vol),
        "post_control_annualized_vol": float(post_control_vol),
        "gmv_score_tilt": float(args.gmv_score_tilt),
        "gmv_penalty_lambda": float(args.gmv_penalty_lambda),
        "allow_cash_vol_control": bool(args.allow_cash_vol_control),
        "full_investment_required": not bool(args.allow_cash_vol_control),
        "max_weight": float(max(daily_weights.values())) if daily_weights else 0.0,
        "min_weight": float(min(daily_weights.values())) if daily_weights else 0.0,
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"State saved: {state_path}")

    # 11. Save predictions ---------------------------------------------------------------------
    pred_dir = Path("outputs/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"latest_predictions_{latest_date}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved: {pred_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
