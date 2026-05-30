"""Generate buy/sell recommendations for the next trading day.

Usage (after daily data sync):
    python scripts/predict_latest.py

This script:
1. Finds the latest available trade date from A股数据/daily/
2. Builds the stock pool for that date
3. Generates predictions using the trained TSN+GRU model
4. Reads current holdings from outputs/current_holdings.csv (if exists)
5. Outputs buy/sell recommendations
"""

from __future__ import annotations

import argparse
import os
import sys
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
    PreprocessConfig,
    build_features_for_date,
    available_daily_dates,
    fill_and_standardize_cross_section,
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

# ---- Helpers ----

def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint.get("model_type", "tsn")
    input_dim = checkpoint["input_dim"]
    hidden_dim = checkpoint.get("hidden_dim", 64)
    dropout = checkpoint.get("dropout", 0.2)

    if model_type == "tsn_gru":
        model = TemporalSegmentGRU(input_dim, hidden_dim=hidden_dim, dropout=dropout)
    else:
        model = TemporalSegmentNet(input_dim, hidden_dim=hidden_dim, dropout=dropout)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily buy/sell recommendations.")
    parser.add_argument("--data-dir", default="A股数据")
    parser.add_argument("--checkpoint", default="outputs/models/tsn_best.pt")
    parser.add_argument("--holdings-csv", default="outputs/current_holdings.csv")
    parser.add_argument("--n-holdings", type=int, default=10)
    parser.add_argument("--max-sell", type=int, default=2)
    parser.add_argument("--buffer-rank", type=int, default=30)
    parser.add_argument("--max-industry-count", type=int, default=3)
    parser.add_argument("--max-daily-volatility", type=float, default=0.08)
    parser.add_argument("--disable-risk-control", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    device = torch.device(args.device)
    enable_risk_control = not args.disable_risk_control

    # 1. Find latest trade date with available data --------------------------------------------------
    trading_dates = available_daily_dates(data_dir)
    if not trading_dates:
        print("ERROR: No daily data found. Please sync data from cloud drive first.")
        return 1

    date_to_index = {date: idx for idx, date in enumerate(trading_dates)}
    latest_date = trading_dates[-1]
    idx = date_to_index[latest_date]
    if idx < args.window:
        print(f"ERROR: Not enough history before {latest_date} (need {args.window} days, got {idx}).")
        return 1

    feature_end_date = trading_dates[idx - 1]
    window_dates = trading_dates[idx - args.window : idx]

    print(f"Latest trade date: {latest_date}")
    print(f"Feature end date:  {feature_end_date}")
    print(f"Window: {window_dates[0]} -> {window_dates[-1]} ({len(window_dates)} days)")
    print()

    # 2. Build stock pool ---------------------------------------------------------------------------
    pool, summary = build_stock_pool(
        data_dir,
        as_of_date=feature_end_date,
        index_code=DEFAULT_INDEX_CODE,
        min_amount=DEFAULT_MIN_AMOUNT,
    )
    print(f"Stock pool: {summary.final_count} stocks (weight date: {summary.weight_trade_date})")

    # 3. Generate sequence features and predictions -------------------------------------------------
    print("Generating sequence features...")
    seq_frame = build_sequence_features_for_date(data_dir, pool, window_dates)
    if seq_frame.empty:
        print("ERROR: No sequence features generated.")
        return 1

    feature_columns = finalized_feature_columns(seq_frame)
    print(f"Feature columns: {len(feature_columns)}")

    # Group by ts_code and build [1, window, features] tensors
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Model checkpoint not found: {checkpoint_path}")
        print("Please run train_tsn.py first.")
        return 1

    model, ckpt = load_model(str(checkpoint_path), device)
    ckpt_features = ckpt.get("feature_columns", feature_columns)
    # Use only features the model was trained on
    usable_features = [c for c in ckpt_features if c in seq_frame.columns]
    if not usable_features:
        print("ERROR: No matching feature columns between data and model checkpoint.")
        return 1

    scores = []
    for ts_code, group in seq_frame.groupby("ts_code", sort=False):
        if group.shape[0] != args.window:
            continue
        group_sorted = group.sort_values("trade_date")
        x = torch.from_numpy(group_sorted[usable_features].to_numpy(dtype=np.float32)).unsqueeze(0)
        x = x.to(device)
        with torch.no_grad():
            score = float(model(x).cpu().item())
        scores.append({"ts_code": ts_code, "score": score})

    if not scores:
        print("ERROR: No predictions generated.")
        return 1

    pred_df = pd.DataFrame(scores).sort_values("score", ascending=False)
    pred_df["trade_date"] = latest_date

    # 4. Load current holdings ---------------------------------------------------------------------
    holdings_path = Path(args.holdings_csv)
    current_holdings = []
    if holdings_path.exists():
        hdf = pd.read_csv(holdings_path, dtype={"ts_code": str})
        current_holdings = hdf["ts_code"].tolist()
        print(f"\nCurrent holdings ({len(current_holdings)}): {current_holdings}")
    else:
        print(f"\nNo holdings file found at {holdings_path} — will treat as initial day.")
        # Create an empty holdings file
        holdings_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["ts_code", "buy_date", "weight"]).to_csv(holdings_path, index=False)

    # 5. Prepare scores and apply risk -------------------------------------------------------------
    day_scores = prepare_daily_scores(pred_df)
    industry_map = read_industry_map(data_dir)
    day_scores = apply_risk_annotations(
        day_scores=day_scores,
        data_dir=data_dir,
        trade_date=latest_date,
        trading_dates=trading_dates,
        date_to_index=date_to_index,
        industry_map=industry_map,
        enable_risk_control=enable_risk_control,
        volatility_window=20,
        max_daily_volatility=args.max_daily_volatility,
    )

    # 6. Determine trades --------------------------------------------------------------------------
    if not current_holdings:
        # Initial build
        buys = select_buy_candidates(
            day_scores=day_scores,
            existing_holdings=[],
            excluded_codes=set(),
            slots=args.n_holdings,
            max_industry_count=args.max_industry_count,
            enable_risk_control=enable_risk_control,
        )
        sells = []
        new_holdings = buys
    else:
        new_holdings, sells, buys = rebalance_holdings(
            holdings=current_holdings,
            day_scores=day_scores,
            n_holdings=args.n_holdings,
            max_sell=args.max_sell,
            buffer_rank=args.buffer_rank,
            strategy="buffered_topn",
            max_industry_count=args.max_industry_count,
            enable_risk_control=enable_risk_control,
        )

    # 7. Print recommendations ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TRADING RECOMMENDATIONS")
    print("=" * 70)

    if sells:
        print(f"\n  SELL ({len(sells)} stocks):")
        for code in sells:
            row = day_scores[day_scores["ts_code"] == code]
            if not row.empty:
                r = row.iloc[0]
                print(f"    {code}  score={r['score']:.6f}  rank={int(r['rank'])}  "
                      f"industry={r['industry']}  risk={r['risk_reason']}")
    else:
        print("\n  SELL: None (initial build or no sells needed)")

    if buys:
        print(f"\n  BUY ({len(buys)} stocks):")
        for code in buys:
            row = day_scores[day_scores["ts_code"] == code]
            if not row.empty:
                r = row.iloc[0]
                print(f"    {code}  score={r['score']:.6f}  rank={int(r['rank'])}  "
                      f"industry={r['industry']}  risk={r['risk_reason']}")
    else:
        print("\n  BUY: None (no changes needed)")

    print(f"\n  FINAL HOLDINGS ({len(new_holdings)} stocks):")
    for i, code in enumerate(new_holdings, 1):
        row = day_scores[day_scores["ts_code"] == code]
        if not row.empty:
            r = row.iloc[0]
            print(f"    {i:2d}. {code}  score={r['score']:.6f}  rank={int(r['rank'])}  "
                  f"industry={r['industry']}")
        else:
            print(f"    {i:2d}. {code}  (no score available)")

    print("\n" + "=" * 70)

    # 8. Save updated holdings ---------------------------------------------------------------------
    holdings_out = pd.DataFrame({
        "ts_code": new_holdings,
        "buy_date": [latest_date] * len(new_holdings),
        "weight": [1.0 / len(new_holdings)] * len(new_holdings),
    })
    holdings_out.to_csv(holdings_path, index=False)
    print(f"\nUpdated holdings saved to: {holdings_path}")

    # 9. Save predictions for reference ------------------------------------------------------------
    pred_dir = Path("outputs/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"latest_predictions_{latest_date}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Full predictions saved to: {pred_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
