from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocess import (
    METRIC_FEATURES,
    RAW_FEATURE_COLUMNS,
    add_base_features,
    add_rolling_features,
    add_technical_features,
    available_daily_dates,
    build_labels,
    fill_and_standardize_cross_section,
    finalized_feature_columns,
    read_market_index,
)
from stock_pool import DEFAULT_INDEX_CODE, DEFAULT_MIN_AMOUNT, build_stock_pool


@dataclass(frozen=True)
class SequencePreprocessConfig:
    data_dir: str = "A股数据"
    output_dir: str = "outputs/preprocessed_seq"
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
class SequencePreprocessSummary:
    start_date: int
    end_date: int
    window: int
    index_code: str
    feature_count: int
    feature_columns: list[str]
    split_rows: dict[str, int]
    split_date_ranges: dict[str, list[int | None]]
    trade_date_count: int
    dropped_incomplete_window: int
    dropped_missing_label: int
    max_pool_size: int
    min_pool_size: int


def build_sequence_dataset(
    config: SequencePreprocessConfig,
) -> tuple[dict[str, np.ndarray], SequencePreprocessSummary]:
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
    samples: list[dict[str, Any]] = []
    feature_columns: list[str] | None = None
    dropped_incomplete_window = 0
    dropped_missing_label = 0
    pool_sizes: list[int] = []

    for trade_date in sample_dates:
        idx = date_to_index[trade_date]
        feature_end_date = trading_dates[idx - 1]
        label_end_date = trading_dates[idx + 1]
        window_dates = trading_dates[idx - config.window : idx]

        # Build multi-horizon label date map
        label_end_dates_map: dict[str, int] = {"1d": label_end_date}
        for horizon, offset in [("3d", 3), ("5d", 5)]:
            future_idx = idx + offset
            if future_idx < len(trading_dates):
                label_end_dates_map[f"{horizon}d"] = trading_dates[future_idx]

        pool, _ = build_stock_pool(
            data_dir,
            as_of_date=feature_end_date,
            index_code=config.index_code,
            min_amount=config.min_amount,
        )
        pool_sizes.append(int(pool.shape[0]))
        if pool.empty:
            continue

        sequence_frame = build_sequence_features_for_date(data_dir, pool, window_dates)
        if sequence_frame.empty:
            dropped_incomplete_window += int(pool.shape[0])
            continue

        if feature_columns is None:
            feature_columns = finalized_feature_columns(sequence_frame)

        labels = build_labels(data_dir, market, trade_date, label_end_dates_map)
        label_map = labels.set_index("ts_code")["label_excess_1d"].to_dict()
        label_map_3d = (
            labels.set_index("ts_code")["label_excess_3d"].to_dict()
            if "label_excess_3d" in labels.columns
            else {}
        )
        label_map_5d = (
            labels.set_index("ts_code")["label_excess_5d"].to_dict()
            if "label_excess_5d" in labels.columns
            else {}
        )
        valid_label_codes = set(labels.dropna(subset=["label_excess_1d"])["ts_code"])

        complete_counts = sequence_frame.groupby("ts_code")["trade_date"].nunique()
        complete_codes = set(complete_counts[complete_counts == config.window].index)
        candidate_codes = [
            code
            for code in pool["ts_code"].astype(str)
            if code in complete_codes and code in valid_label_codes
        ]
        dropped_incomplete_window += int(pool.shape[0] - len(complete_codes))
        dropped_missing_label += int(len(complete_codes - valid_label_codes))

        meta = pool.set_index("ts_code")
        grouped = {
            code: frame.sort_values("trade_date")
            for code, frame in sequence_frame.groupby("ts_code", sort=False)
            if code in candidate_codes
        }
        for code in candidate_codes:
            frame = grouped.get(code)
            if frame is None or frame.shape[0] != config.window:
                continue
            samples.append(
                {
                    "X": frame[feature_columns].to_numpy(dtype=np.float32),
                    "y": np.float32(label_map.get(code, np.nan)),
                    "y_3d": np.float32(label_map_3d.get(code, np.nan)),
                    "y_5d": np.float32(label_map_5d.get(code, np.nan)),
                    "trade_date": np.int32(trade_date),
                    "feature_end_date": np.int32(feature_end_date),
                    "label_end_date": np.int32(label_end_date),
                    "feature_dates": np.asarray(window_dates, dtype=np.int32),
                    "ts_code": code,
                    "weight": np.float32(meta.at[code, "weight"]) if "weight" in meta else np.float32(0.0),
                    "industry": str(meta.at[code, "industry"]) if "industry" in meta else "",
                }
            )

    if not samples:
        raise ValueError("Sequence preprocessing produced no rows")
    assert feature_columns is not None

    arrays = pack_samples(samples)
    splits = split_sequence_arrays(arrays, config)
    summary = SequencePreprocessSummary(
        start_date=config.start_date,
        end_date=end_date,
        window=config.window,
        index_code=config.index_code,
        feature_count=len(feature_columns),
        feature_columns=feature_columns,
        split_rows={name: int(split["y"].shape[0]) for name, split in splits.items()},
        split_date_ranges=split_date_ranges(splits),
        trade_date_count=int(np.unique(arrays["trade_date"]).shape[0]),
        dropped_incomplete_window=int(dropped_incomplete_window),
        dropped_missing_label=int(dropped_missing_label),
        max_pool_size=max(pool_sizes) if pool_sizes else 0,
        min_pool_size=min(pool_sizes) if pool_sizes else 0,
    )
    for split in splits.values():
        split["feature_columns"] = np.asarray(feature_columns, dtype=object)
    return splits, summary


def build_sequence_features_for_date(
    data_dir: Path,
    pool: pd.DataFrame,
    window_dates: list[int],
) -> pd.DataFrame:
    pool_codes = set(pool["ts_code"].astype(str))
    frames = [
        read_enriched_day(str(data_dir.resolve()), date, tuple(sorted(pool_codes)))
        for date in window_dates
    ]
    history = pd.concat(frames, ignore_index=True)
    if history.empty:
        return pd.DataFrame()

    history = history.sort_values(["ts_code", "trade_date"])
    history = add_base_features(history)
    history = add_rolling_features(history)
    history = add_technical_features(history)

    standardized_days: list[pd.DataFrame] = []
    for date in window_dates:
        day = history[history["trade_date"] == date].copy()
        if day.empty:
            continue
        day = fill_and_standardize_cross_section(day)
        standardized_days.append(day[["ts_code", "trade_date", *finalized_feature_columns(day)]])
    if not standardized_days:
        return pd.DataFrame()
    return pd.concat(standardized_days, ignore_index=True)


@lru_cache(maxsize=128)
def read_enriched_day(
    data_dir: str,
    date: int,
    pool_codes: tuple[str, ...],
) -> pd.DataFrame:
    root = Path(data_dir)
    pool_set = set(pool_codes)
    daily_columns = [
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
    day = pd.read_csv(
        root / "daily" / f"{date}.csv",
        usecols=daily_columns,
        dtype={"ts_code": str},
    )
    day = day[day["ts_code"].isin(pool_set)].copy()

    metric = pd.read_csv(
        root / "metric" / f"{date}.csv",
        usecols=["ts_code", *METRIC_FEATURES],
        dtype={"ts_code": str},
    )
    moneyflow = pd.read_csv(
        root / "moneyflow" / f"{date}.csv",
        usecols=[
            "ts_code",
            "buy_lg_amount",
            "sell_lg_amount",
            "buy_elg_amount",
            "sell_elg_amount",
            "net_mf_amount",
        ],
        dtype={"ts_code": str},
    )
    return day.merge(metric, on="ts_code", how="left").merge(
        moneyflow,
        on="ts_code",
        how="left",
    )


def pack_samples(samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "X": np.stack([sample["X"] for sample in samples]).astype(np.float32),
        "y": np.asarray([sample["y"] for sample in samples], dtype=np.float32),
        "y_3d": np.asarray([sample["y_3d"] for sample in samples], dtype=np.float32),
        "y_5d": np.asarray([sample["y_5d"] for sample in samples], dtype=np.float32),
        "trade_date": np.asarray([sample["trade_date"] for sample in samples], dtype=np.int32),
        "feature_end_date": np.asarray(
            [sample["feature_end_date"] for sample in samples],
            dtype=np.int32,
        ),
        "label_end_date": np.asarray(
            [sample["label_end_date"] for sample in samples],
            dtype=np.int32,
        ),
        "feature_dates": np.stack([sample["feature_dates"] for sample in samples]).astype(np.int32),
        "ts_code": np.asarray([sample["ts_code"] for sample in samples], dtype=object),
        "weight": np.asarray([sample["weight"] for sample in samples], dtype=np.float32),
        "industry": np.asarray([sample["industry"] for sample in samples], dtype=object),
    }


def split_sequence_arrays(
    arrays: dict[str, np.ndarray],
    config: SequencePreprocessConfig,
) -> dict[str, dict[str, np.ndarray]]:
    dates = arrays["trade_date"]
    masks = {
        "train": dates <= config.train_end,
        "valid": (dates >= config.valid_start) & (dates <= config.valid_end),
    }
    if config.test_start is not None:
        masks["test"] = dates >= config.test_start
    return {
        name: {key: value[mask] for key, value in arrays.items()}
        for name, mask in masks.items()
    }


def split_date_ranges(splits: dict[str, dict[str, np.ndarray]]) -> dict[str, list[int | None]]:
    ranges: dict[str, list[int | None]] = {}
    for name, split in splits.items():
        dates = split["trade_date"]
        if dates.size == 0:
            ranges[name] = [None, None]
        else:
            ranges[name] = [int(dates.min()), int(dates.max())]
    return ranges


def write_sequence_outputs(
    splits: dict[str, dict[str, np.ndarray]],
    summary: SequencePreprocessSummary,
    config: SequencePreprocessConfig,
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split in splits.items():
        np.savez_compressed(output_dir / f"sequences_{split_name}.npz", **split)
    (output_dir / "sequence_summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

