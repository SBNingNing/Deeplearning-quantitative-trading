from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_pool import DEFAULT_INDEX_CODE, DEFAULT_MIN_AMOUNT, build_stock_pool


PRICE_FEATURES = [
    "ret_1d",
    "amplitude",
    "gap_open",
    "close_pos",
    "vol_chg",
    "amount_chg",
    "vwap_deviation",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "ma5_deviation",
    "ma10_deviation",
    "ma20_deviation",
    "amount_mean_5",
    "amount_mean_10",
    "amount_mean_20",
]

METRIC_FEATURES = [
    "turnover_rate_f",
    "volume_ratio",
    "pe_ttm",
    "pb",
    "ps_ttm",
    "dv_ttm",
    "total_mv",
    "circ_mv",
]

MONEYFLOW_FEATURES = [
    "net_mf_amount_ratio",
    "big_order_net_amount_ratio",
    "elg_order_net_amount_ratio",
    "net_mf_amount_ratio_mean_5",
    "net_mf_amount_ratio_mean_10",
    "big_order_net_amount_ratio_mean_5",
    "big_order_net_amount_ratio_mean_10",
]

RAW_FEATURE_COLUMNS = PRICE_FEATURES + METRIC_FEATURES + MONEYFLOW_FEATURES
ID_COLUMNS = [
    "trade_date",
    "feature_end_date",
    "label_end_date",
    "ts_code",
    "weight",
    "industry",
]
LABEL_COLUMNS = ["label_return_1d", "label_index_return_1d", "label_excess_1d"]


@dataclass(frozen=True)
class PreprocessConfig:
    data_dir: str = "A股数据"
    output_dir: str = "outputs/preprocessed"
    index_code: str = DEFAULT_INDEX_CODE
    start_date: int = 20190101
    end_date: int | None = None
    window: int = 20
    min_amount: float = DEFAULT_MIN_AMOUNT
    train_end: int = 20241231
    valid_start: int = 20250101
    valid_end: int = 20251231
    test_start: int = 20260101


@dataclass(frozen=True)
class PreprocessSummary:
    start_date: int
    end_date: int
    window: int
    index_code: str
    feature_count: int
    feature_columns: list[str]
    split_rows: dict[str, int]
    split_date_ranges: dict[str, list[int | None]]
    trade_date_count: int
    dropped_missing_label: int
    max_pool_size: int
    min_pool_size: int
    missing_rate: dict[str, float]


def build_preprocessed_dataset(config: PreprocessConfig) -> tuple[pd.DataFrame, PreprocessSummary]:
    data_dir = Path(config.data_dir)
    trading_dates = available_daily_dates(data_dir)
    if not trading_dates:
        raise FileNotFoundError(f"No daily files found under {data_dir / 'daily'}")

    end_date = config.end_date or trading_dates[-2]
    date_to_index = {date: index for index, date in enumerate(trading_dates)}
    sample_dates = [
        date
        for date in trading_dates
        if config.start_date <= date <= end_date
        and date_to_index[date] >= config.window
        and date_to_index[date] + 1 < len(trading_dates)
    ]

    if not sample_dates:
        raise ValueError("No sample dates available for the requested range")

    market = read_market_index(data_dir, config.index_code)
    frames: list[pd.DataFrame] = []
    pool_sizes: list[int] = []
    dropped_missing_label = 0

    for trade_date in sample_dates:
        idx = date_to_index[trade_date]
        feature_end_date = trading_dates[idx - 1]
        label_end_date = trading_dates[idx + 1]
        window_dates = trading_dates[idx - config.window : idx]

        pool, _ = build_stock_pool(
            data_dir,
            as_of_date=feature_end_date,
            index_code=config.index_code,
            min_amount=config.min_amount,
        )
        pool_sizes.append(int(pool.shape[0]))
        if pool.empty:
            continue

        features = build_features_for_date(data_dir, pool, window_dates)
        labels = build_labels(data_dir, market, trade_date, label_end_date)
        rows = features.merge(labels, on="ts_code", how="left")
        before_label_filter = len(rows)
        rows = rows.dropna(subset=["label_return_1d", "label_excess_1d"]).copy()
        dropped_missing_label += before_label_filter - len(rows)
        if rows.empty:
            continue

        rows.insert(0, "label_end_date", label_end_date)
        rows.insert(0, "feature_end_date", feature_end_date)
        rows.insert(0, "trade_date", trade_date)
        rows = fill_and_standardize_cross_section(rows)
        frames.append(rows)

    if not frames:
        raise ValueError("Preprocessing produced no rows")

    dataset = pd.concat(frames, ignore_index=True)
    feature_columns = finalized_feature_columns(dataset)
    dataset = dataset[ID_COLUMNS + feature_columns + LABEL_COLUMNS]
    split_rows = split_row_counts(dataset, config)
    split_date_ranges = split_ranges(dataset, config)
    missing_rate = dataset[feature_columns].isna().mean().round(6).to_dict()

    summary = PreprocessSummary(
        start_date=config.start_date,
        end_date=end_date,
        window=config.window,
        index_code=config.index_code,
        feature_count=len(feature_columns),
        feature_columns=feature_columns,
        split_rows=split_rows,
        split_date_ranges=split_date_ranges,
        trade_date_count=int(dataset["trade_date"].nunique()),
        dropped_missing_label=int(dropped_missing_label),
        max_pool_size=max(pool_sizes) if pool_sizes else 0,
        min_pool_size=min(pool_sizes) if pool_sizes else 0,
        missing_rate={key: float(value) for key, value in missing_rate.items()},
    )
    return dataset, summary


def write_preprocessed_outputs(
    dataset: pd.DataFrame,
    summary: PreprocessSummary,
    config: PreprocessConfig,
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in split_dataset(dataset, config).items():
        split_df.to_parquet(output_dir / f"features_{split_name}.parquet", index=False)

    summary_path = output_dir / "preprocess_summary.json"
    summary_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_features_for_date(
    data_dir: Path,
    pool: pd.DataFrame,
    window_dates: list[int],
) -> pd.DataFrame:
    pool_codes = set(pool["ts_code"])
    frames: list[pd.DataFrame] = []
    for date in window_dates:
        day = read_daily_features(data_dir, date, pool_codes)
        day = merge_metric_features(data_dir, date, day)
        day = merge_moneyflow_features(data_dir, date, day)
        frames.append(day)

    history = pd.concat(frames, ignore_index=True)
    history = history.sort_values(["ts_code", "trade_date"])
    history = add_base_features(history)
    history = add_rolling_features(history)
    latest_date = window_dates[-1]
    latest = history[history["trade_date"] == latest_date].copy()
    latest = latest.merge(
        pool[["ts_code", "weight", "industry"]],
        on="ts_code",
        how="left",
    )
    return latest[["ts_code", "weight", "industry", *RAW_FEATURE_COLUMNS]]


def read_daily_features(data_dir: Path, date: int, pool_codes: set[str]) -> pd.DataFrame:
    path = data_dir / "daily" / f"{date}.csv"
    columns = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "vol",
        "amount",
        "vwap",
    ]
    df = pd.read_csv(path, usecols=columns, dtype={"ts_code": str})
    return df[df["ts_code"].isin(pool_codes)].copy()


def merge_metric_features(data_dir: Path, date: int, base: pd.DataFrame) -> pd.DataFrame:
    path = data_dir / "metric" / f"{date}.csv"
    columns = ["ts_code", *METRIC_FEATURES]
    metric = pd.read_csv(path, usecols=columns, dtype={"ts_code": str})
    return base.merge(metric, on="ts_code", how="left")


def merge_moneyflow_features(data_dir: Path, date: int, base: pd.DataFrame) -> pd.DataFrame:
    path = data_dir / "moneyflow" / f"{date}.csv"
    columns = [
        "ts_code",
        "buy_lg_amount",
        "sell_lg_amount",
        "buy_elg_amount",
        "sell_elg_amount",
        "net_mf_amount",
    ]
    moneyflow = pd.read_csv(path, usecols=columns, dtype={"ts_code": str})
    return base.merge(moneyflow, on="ts_code", how="left")


def add_base_features(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "vol",
        "amount",
        "vwap",
        *METRIC_FEATURES,
        "buy_lg_amount",
        "sell_lg_amount",
        "buy_elg_amount",
        "sell_elg_amount",
        "net_mf_amount",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["ret_1d"] = safe_divide(df["close"], df["pre_close"]) - 1.0
    df["amplitude"] = safe_divide(df["high"], df["low"]) - 1.0
    df["gap_open"] = safe_divide(df["open"], df["pre_close"]) - 1.0
    df["close_pos"] = safe_divide(df["close"] - df["low"], df["high"] - df["low"])
    df["vwap_deviation"] = safe_divide(df["close"], df["vwap"]) - 1.0
    df["vol_chg"] = df.groupby("ts_code")["vol"].pct_change()
    df["amount_chg"] = df.groupby("ts_code")["amount"].pct_change()

    amount_wan = df["amount"] / 10.0
    big_net = (
        df["buy_lg_amount"]
        + df["buy_elg_amount"]
        - df["sell_lg_amount"]
        - df["sell_elg_amount"]
    )
    elg_net = df["buy_elg_amount"] - df["sell_elg_amount"]
    df["net_mf_amount_ratio"] = safe_divide(df["net_mf_amount"], amount_wan)
    df["big_order_net_amount_ratio"] = safe_divide(big_net, amount_wan)
    df["elg_order_net_amount_ratio"] = safe_divide(elg_net, amount_wan)

    for column in ["pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv"]:
        df[column] = np.log1p(df[column].clip(lower=0))
    return df


def add_rolling_features(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    grouped = df.groupby("ts_code", group_keys=False)
    for window in [5, 10, 20]:
        df[f"momentum_{window}"] = grouped["close"].transform(
            lambda values: values / values.shift(window - 1) - 1.0
        )
        df[f"volatility_{window}"] = grouped["ret_1d"].transform(
            lambda values: values.rolling(window, min_periods=max(3, window // 2)).std()
        )
        ma = grouped["close"].transform(
            lambda values: values.rolling(window, min_periods=max(3, window // 2)).mean()
        )
        df[f"ma{window}_deviation"] = safe_divide(df["close"], ma) - 1.0
        df[f"amount_mean_{window}"] = grouped["amount"].transform(
            lambda values: values.rolling(window, min_periods=max(3, window // 2)).mean()
        )

    for column in ["net_mf_amount_ratio", "big_order_net_amount_ratio"]:
        for window in [5, 10]:
            df[f"{column}_mean_{window}"] = grouped[column].transform(
                lambda values: values.rolling(
                    window,
                    min_periods=max(3, window // 2),
                ).mean()
            )
    return df


def build_labels(
    data_dir: Path,
    market: pd.DataFrame,
    trade_date: int,
    label_end_date: int,
) -> pd.DataFrame:
    today = pd.read_csv(
        data_dir / "daily" / f"{trade_date}.csv",
        usecols=["ts_code", "close"],
        dtype={"ts_code": str},
    ).rename(columns={"close": "close_t"})
    next_day = pd.read_csv(
        data_dir / "daily" / f"{label_end_date}.csv",
        usecols=["ts_code", "close"],
        dtype={"ts_code": str},
    ).rename(columns={"close": "close_t1"})
    labels = today.merge(next_day, on="ts_code", how="inner")
    labels["label_return_1d"] = safe_divide(labels["close_t1"], labels["close_t"]) - 1.0

    index_today = market.loc[market["trade_date"] == trade_date, "close"]
    index_next = market.loc[market["trade_date"] == label_end_date, "close"]
    if index_today.empty or index_next.empty:
        index_return = np.nan
    else:
        index_return = float(index_next.iloc[0] / index_today.iloc[0] - 1.0)
    labels["label_index_return_1d"] = index_return
    labels["label_excess_1d"] = labels["label_return_1d"] - index_return
    return labels[["ts_code", *LABEL_COLUMNS]]


def fill_and_standardize_cross_section(rows: pd.DataFrame) -> pd.DataFrame:
    df = rows.copy()
    derived: dict[str, pd.Series | float] = {}
    for column in RAW_FEATURE_COLUMNS:
        missing_col = f"{column}_missing"
        derived[missing_col] = df[column].isna().astype("int8")
        if df[column].notna().any():
            low = df[column].quantile(0.01)
            high = df[column].quantile(0.99)
            df[column] = df[column].clip(low, high)
            fill_value = df[column].median()
        else:
            fill_value = 0.0
        df[column] = df[column].fillna(fill_value)

        std = df[column].std(ddof=0)
        if pd.isna(std) or std == 0:
            derived[f"{column}_z"] = pd.Series(0.0, index=df.index)
        else:
            derived[f"{column}_z"] = (df[column] - df[column].mean()) / std
        derived[f"{column}_rank"] = df[column].rank(pct=True)
    return pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)


def finalized_feature_columns(dataset: pd.DataFrame) -> list[str]:
    suffixes = ("_z", "_rank", "_missing")
    columns: list[str] = []
    for column in RAW_FEATURE_COLUMNS:
        columns.extend(
            candidate
            for candidate in [column, f"{column}_z", f"{column}_rank", f"{column}_missing"]
            if candidate in dataset.columns
        )
    return [column for column in columns if column.endswith(suffixes) or column in RAW_FEATURE_COLUMNS]


def split_dataset(dataset: pd.DataFrame, config: PreprocessConfig) -> dict[str, pd.DataFrame]:
    train = dataset[dataset["trade_date"] <= config.train_end].copy()
    valid = dataset[
        (dataset["trade_date"] >= config.valid_start)
        & (dataset["trade_date"] <= config.valid_end)
    ].copy()
    test = dataset[dataset["trade_date"] >= config.test_start].copy()
    return {"train": train, "valid": valid, "test": test}


def split_row_counts(dataset: pd.DataFrame, config: PreprocessConfig) -> dict[str, int]:
    return {name: int(frame.shape[0]) for name, frame in split_dataset(dataset, config).items()}


def split_ranges(dataset: pd.DataFrame, config: PreprocessConfig) -> dict[str, list[int | None]]:
    ranges: dict[str, list[int | None]] = {}
    for name, frame in split_dataset(dataset, config).items():
        if frame.empty:
            ranges[name] = [None, None]
        else:
            ranges[name] = [int(frame["trade_date"].min()), int(frame["trade_date"].max())]
    return ranges


def read_market_index(data_dir: Path, index_code: str) -> pd.DataFrame:
    path = data_dir / "market" / f"{index_code}.csv"
    df = pd.read_csv(path, usecols=["trade_date", "close"])
    df["trade_date"] = pd.to_numeric(df["trade_date"], errors="coerce").astype("Int64")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna().astype({"trade_date": int})


def available_daily_dates(data_dir: Path) -> list[int]:
    daily_dir = data_dir / "daily"
    return sorted(
        int(path.stem)
        for path in daily_dir.glob("*.csv")
        if path.stem.isdigit() and len(path.stem) == 8
    )


def safe_divide(numerator: Any, denominator: Any) -> Any:
    result = numerator / denominator
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan)
    if np.isinf(result):
        return np.nan
    return result

