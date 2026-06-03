"""Factor IC analysis for raw features.

Computes daily cross-sectional Spearman Rank IC between each raw factor and a
forward excess-return label. The report keeps the signed IC direction and adds
t-statistics, p-values, and positive-day ratios so the output is easier to use
as a factor validation table.

Usage:
    python scripts/factor_analysis.py
    python scripts/factor_analysis.py --split valid
    python scripts/factor_analysis.py --split valid --label-horizon 5d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocess import RAW_FEATURE_COLUMNS  # noqa: E402


def compute_daily_rank_ic(
    feature_values: np.ndarray,
    returns: np.ndarray,
    dates: np.ndarray,
    min_pairs: int = 10,
) -> dict[str, float]:
    """Compute daily cross-sectional rank IC stats for a single factor.

    ICIR is the daily mean/std ratio. The t-statistic is computed as
    mean / (std / sqrt(n_days)), with a two-sided t-test p-value.
    """
    ics: list[float] = []
    for date in np.unique(dates):
        mask = dates == date
        fv = feature_values[mask]
        rt = returns[mask]
        valid = np.isfinite(fv) & np.isfinite(rt)
        if valid.sum() < min_pairs:
            continue

        fv_valid = fv[valid]
        rt_valid = rt[valid]
        if np.std(fv_valid) == 0 or np.std(rt_valid) == 0:
            continue

        ic = pd.Series(fv_valid).corr(pd.Series(rt_valid), method="spearman")
        if pd.notna(ic):
            ics.append(float(ic))

    if not ics:
        return {
            "mean_ic": float("nan"),
            "median_ic": float("nan"),
            "ic_std": float("nan"),
            "icir": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "positive_rate": float("nan"),
            "abs_mean_ic": float("nan"),
            "n_days": 0,
        }

    ic_array = np.asarray(ics, dtype=float)
    mean_ic = float(np.mean(ic_array))
    median_ic = float(np.median(ic_array))
    ic_std = float(np.std(ic_array, ddof=1)) if len(ic_array) > 1 else 0.0
    icir = mean_ic / ic_std if ic_std > 0 else float("nan")
    t_stat = icir * np.sqrt(len(ic_array)) if ic_std > 0 else float("nan")
    p_value = (
        float(2.0 * stats.t.sf(abs(t_stat), df=len(ic_array) - 1))
        if len(ic_array) > 1 and np.isfinite(t_stat)
        else float("nan")
    )

    return {
        "mean_ic": round(mean_ic, 6),
        "median_ic": round(median_ic, 6),
        "ic_std": round(ic_std, 6),
        "icir": round(icir, 4) if np.isfinite(icir) else float("nan"),
        "t_stat": round(float(t_stat), 4) if np.isfinite(t_stat) else float("nan"),
        "p_value": round(p_value, 6) if np.isfinite(p_value) else float("nan"),
        "positive_rate": round(float(np.mean(ic_array > 0.0)), 6),
        "abs_mean_ic": round(abs(mean_ic), 6),
        "n_days": int(len(ic_array)),
    }


def label_key_from_horizon(label_horizon: str) -> str:
    """Map CLI horizon names to keys stored in sequence npz files."""
    if label_horizon == "1d":
        return "y"
    if label_horizon in {"3d", "5d"}:
        return f"y_{label_horizon}"
    return label_horizon


def analyze_factors(
    data_dir: str,
    split: str,
    label_horizon: str,
    min_pairs: int,
) -> pd.DataFrame:
    """Run single-factor IC analysis on the specified data split."""
    data_path = Path(data_dir) / f"sequences_{split}.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    data = np.load(data_path, allow_pickle=True)
    label_key = label_key_from_horizon(label_horizon)
    if label_key not in data:
        raise KeyError(f"Label '{label_key}' not found in {data_path}")

    X = data["X"]  # [N, window, F]
    y = data[label_key]  # [N]
    if np.isfinite(y).sum() < min_pairs:
        raise ValueError(
            f"Label '{label_key}' in {data_path} has fewer than {min_pairs} "
            "finite values. Regenerate the sequence dataset if this is a "
            "newly enabled horizon."
        )
    trade_dates = data["trade_date"]  # [N]
    feature_cols = list(data["feature_columns"])

    raw_features = [
        c
        for c in feature_cols
        if c in RAW_FEATURE_COLUMNS
        and not c.endswith("_z")
        and not c.endswith("_rank")
        and not c.endswith("_missing")
    ]

    print(
        f"Analyzing {len(raw_features)} raw factors on {len(y)} samples "
        f"({len(np.unique(trade_dates))} trading days), label={label_horizon}"
    )

    # Use the last time step of each sample's feature window.
    X_last = X[:, -1, :]

    results = []
    for feat_name in raw_features:
        idx = feature_cols.index(feat_name)
        stats_row = compute_daily_rank_ic(
            X_last[:, idx],
            y,
            trade_dates,
            min_pairs=min_pairs,
        )
        results.append({"factor": feat_name, **stats_row})

    return pd.DataFrame(results).sort_values("abs_mean_ic", ascending=False)


def factor_correlation(
    data_dir: str,
    split: str,
    factor_rank: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """Compute Spearman correlation matrix of top-N usable factors by |IC|."""
    data_path = Path(data_dir) / f"sequences_{split}.npz"
    data = np.load(data_path, allow_pickle=True)
    X = data["X"]
    feature_cols = list(data["feature_columns"])

    top = factor_rank.dropna(subset=["abs_mean_ic"])["factor"].head(top_n).tolist()
    top_indices = [feature_cols.index(f) for f in top]
    top_data = pd.DataFrame(X[:, -1, top_indices], columns=top)
    return top_data.corr(method="spearman")


def add_factor_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Attach human-readable factor categories."""
    from preprocess import (  # noqa: E402
        METRIC_FEATURES,
        MONEYFLOW_FEATURES,
        PRICE_FEATURES,
        TECHNICAL_FEATURES,
    )

    category = {}
    for factor in PRICE_FEATURES:
        category[factor] = "Price"
    for factor in TECHNICAL_FEATURES:
        category[factor] = "Technical"
    for factor in METRIC_FEATURES:
        category[factor] = "Fundamental"
    for factor in MONEYFLOW_FEATURES:
        category[factor] = "MoneyFlow"

    out = df.copy()
    out["category"] = out["factor"].map(category).fillna("Other")
    return out


def print_factor_table(df: pd.DataFrame) -> None:
    """Pretty-print the IC table grouped by factor category."""
    df = add_factor_categories(df)

    print("\n" + "=" * 112)
    print("  SINGLE-FACTOR RANK IC ANALYSIS")
    print("=" * 112)
    print(
        f"  {'Factor':<32s} {'Cat':<12s} {'IC':>8s} {'|IC|':>8s} "
        f"{'ICIR':>8s} {'t':>8s} {'p':>9s} {'Pos%':>7s} {'Days':>6s}"
    )
    print("  " + "-" * 109)
    for _, row in df.iterrows():
        mean_ic = f"{row['mean_ic']:.4f}" if not pd.isna(row["mean_ic"]) else "   nan"
        abs_ic = f"{row['abs_mean_ic']:.4f}" if not pd.isna(row["abs_mean_ic"]) else "   nan"
        icir = f"{row['icir']:.2f}" if not pd.isna(row["icir"]) else "   nan"
        t_stat = f"{row['t_stat']:.2f}" if not pd.isna(row["t_stat"]) else "   nan"
        p_value = f"{row['p_value']:.4f}" if not pd.isna(row["p_value"]) else "    nan"
        pos = (
            f"{row['positive_rate'] * 100:.1f}"
            if not pd.isna(row["positive_rate"])
            else "  nan"
        )
        print(
            f"  {row['factor']:<32s} {row['category']:<12s} {mean_ic:>8s} "
            f"{abs_ic:>8s} {icir:>8s} {t_stat:>8s} {p_value:>9s} "
            f"{pos:>7s} {int(row['n_days']):>6d}"
        )

    print("\n  Category Summary:")
    for cat_name, cat_df in df.groupby("category"):
        mean_ic = cat_df["mean_ic"].mean()
        mean_abs_ic = cat_df["abs_mean_ic"].mean()
        count = len(cat_df)
        print(
            f"    {cat_name:<12s}: mean IC={mean_ic: .4f}, "
            f"mean |IC|={mean_abs_ic:.4f}  ({count} factors)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-factor IC analysis.")
    parser.add_argument("--data-dir", default="outputs/preprocessed_seq")
    parser.add_argument(
        "--split",
        default="valid",
        choices=["train", "valid", "test"],
        help="Data split to analyze.",
    )
    parser.add_argument(
        "--label-horizon",
        default="1d",
        choices=["1d", "3d", "5d"],
        help="Forward excess-return horizon to analyze.",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=10,
        help="Minimum valid stocks per day required for IC.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Two-sided p-value threshold for significance.",
    )
    parser.add_argument("--output-dir", default="outputs/factor_analysis")
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top usable factors for correlation matrix.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Loading {args.split} split from {args.data_dir} ...")
    ic_table = analyze_factors(
        args.data_dir,
        args.split,
        label_horizon=args.label_horizon,
        min_pairs=args.min_pairs,
    )
    ic_table = add_factor_categories(ic_table)
    print_factor_table(ic_table)

    print(f"\nComputing correlation matrix for top {args.top_n} usable factors ...")
    corr = factor_correlation(args.data_dir, args.split, ic_table, top_n=args.top_n)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.split if args.label_horizon == "1d" else f"{args.split}_{args.label_horizon}"
    ic_path = output_dir / f"factor_ic_{suffix}.csv"
    corr_path = (
        output_dir / "factor_correlation.csv"
        if args.label_horizon == "1d"
        else output_dir / f"factor_correlation_{suffix}.csv"
    )
    ic_table.to_csv(ic_path, index=False)
    corr.to_csv(corr_path)
    print(f"\nSaved: {ic_path}")
    print(f"Saved: {corr_path}")

    weak_signal = int((ic_table["abs_mean_ic"] > 0.005).sum())
    statistically_significant = int(
        ((ic_table["p_value"] < args.alpha) & (ic_table["n_days"] > 0)).sum()
    )
    usable = int((ic_table["n_days"] > 0).sum())
    positive = int((ic_table["mean_ic"] > 0).sum())

    print(f"\nUsable factors: {usable}/{len(ic_table)}")
    print(f"Factors with |IC| > 0.005: {weak_signal}/{len(ic_table)}")
    print(f"Factors with p < {args.alpha:g}: {statistically_significant}/{len(ic_table)}")
    print(f"Factors with positive IC: {positive}/{len(ic_table)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
