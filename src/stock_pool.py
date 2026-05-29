from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_INDEX_CODE = "000300.SH"
DEFAULT_MIN_AMOUNT = 10_000.0  # thousand yuan, i.e. 10 million yuan
VALID_MARKETS = {"主板", "创业板", "科创板"}
REQUIRED_DAILY_COLUMNS = ["open", "high", "low", "close", "vol", "amount"]


@dataclass(frozen=True)
class WeightSnapshot:
    path: Path
    trade_date: int
    index_code: str
    row_count: int


@dataclass(frozen=True)
class StockPoolSummary:
    as_of_date: int
    index_code: str
    weight_file: str
    weight_trade_date: int
    index_member_count: int
    eligible_market_count: int
    daily_available_count: int
    st_excluded_count: int
    final_count: int
    min_amount: float


def latest_daily_date(data_dir: str | Path) -> int:
    daily_dir = Path(data_dir) / "daily"
    dates = _csv_dates(daily_dir)
    if not dates:
        raise FileNotFoundError(f"No daily csv files found in {daily_dir}")
    return max(dates)


def build_stock_pool(
    data_dir: str | Path,
    as_of_date: int | str | None = None,
    index_code: str = DEFAULT_INDEX_CODE,
    min_amount: float = DEFAULT_MIN_AMOUNT,
    strict_weight_before_date: bool = True,
) -> tuple[pd.DataFrame, StockPoolSummary]:
    """Build an as-of stock pool for index-enhancement experiments.

    The returned pool only uses an index-weight snapshot whose trade_date is
    before the as_of_date by default. This avoids leaking future constituents
    into historical samples and backtests.
    """
    root = Path(data_dir)
    date = int(as_of_date) if as_of_date is not None else latest_daily_date(root)

    basic = read_basic(root)
    daily = read_daily(root, date)
    st_codes = read_st_codes(root, date)
    weights, snapshot = read_latest_index_weights(
        root,
        date,
        index_code=index_code,
        strict_before_date=strict_weight_before_date,
    )

    eligible_basic = basic[basic["market"].isin(VALID_MARKETS)].copy()
    valid_daily = filter_valid_daily(daily, min_amount=min_amount)

    weights_for_merge = weights.rename(
        columns={"con_code": "ts_code", "trade_date": "weight_trade_date"}
    )
    pre_st_pool = (
        weights_for_merge
        .merge(eligible_basic, on="ts_code", how="inner")
        .merge(valid_daily, on="ts_code", how="inner")
    )

    pool = pre_st_pool
    if st_codes:
        pool = pool[~pool["ts_code"].isin(st_codes)].copy()

    keep_columns = [
        "ts_code",
        "name",
        "market",
        "industry",
        "trade_date",
        "weight_trade_date",
        "weight",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "pct_chg",
        "vol",
        "amount",
        "vwap",
    ]
    existing_columns = [column for column in keep_columns if column in pool.columns]
    pool = pool[existing_columns].sort_values(
        ["weight", "amount"], ascending=[False, False]
    )
    pool = pool.reset_index(drop=True)

    summary = StockPoolSummary(
        as_of_date=date,
        index_code=index_code,
        weight_file=str(snapshot.path),
        weight_trade_date=snapshot.trade_date,
        index_member_count=snapshot.row_count,
        eligible_market_count=int(
            weights_for_merge.merge(
                eligible_basic[["ts_code"]], on="ts_code", how="inner"
            ).shape[0]
        ),
        daily_available_count=int(
            weights_for_merge.merge(
                valid_daily[["ts_code"]], on="ts_code", how="inner"
            ).shape[0]
        ),
        st_excluded_count=int(pre_st_pool["ts_code"].isin(st_codes).sum()),
        final_count=int(pool.shape[0]),
        min_amount=float(min_amount),
    )
    return pool, summary


def read_basic(data_dir: str | Path) -> pd.DataFrame:
    path = Path(data_dir) / "basic.csv"
    df = pd.read_csv(path, dtype={"ts_code": str, "market": str})
    _require_columns(df, ["ts_code", "market"], path)
    return df


def read_daily(data_dir: str | Path, trade_date: int | str) -> pd.DataFrame:
    path = Path(data_dir) / "daily" / f"{int(trade_date)}.csv"
    df = pd.read_csv(path, dtype={"ts_code": str})
    _require_columns(df, ["ts_code", "trade_date", *REQUIRED_DAILY_COLUMNS], path)
    return df


def filter_valid_daily(daily: pd.DataFrame, min_amount: float) -> pd.DataFrame:
    df = daily.copy()
    for column in REQUIRED_DAILY_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    mask = df[REQUIRED_DAILY_COLUMNS].notna().all(axis=1)
    mask &= (df[["open", "high", "low", "close"]] > 0).all(axis=1)
    mask &= df["vol"] > 0
    mask &= df["amount"] >= min_amount
    return df[mask].copy()


def read_latest_index_weights(
    data_dir: str | Path,
    as_of_date: int | str,
    index_code: str = DEFAULT_INDEX_CODE,
    strict_before_date: bool = True,
) -> tuple[pd.DataFrame, WeightSnapshot]:
    root = Path(data_dir)
    index_dir = root / "index_weight"
    cutoff = int(as_of_date)
    candidates: list[tuple[int, Path, pd.DataFrame]] = []

    for path in sorted(index_dir.glob(f"*_{index_code}.csv")):
        df = pd.read_csv(path, dtype={"index_code": str, "con_code": str})
        if df.empty:
            continue
        _require_columns(df, ["index_code", "con_code", "trade_date", "weight"], path)

        df = df[df["index_code"] == index_code].copy()
        if df.empty:
            continue
        df["trade_date"] = pd.to_numeric(df["trade_date"], errors="coerce")
        df = df.dropna(subset=["trade_date", "con_code", "weight"])
        if df.empty:
            continue

        date_mask = df["trade_date"] < cutoff
        if not strict_before_date:
            date_mask = df["trade_date"] <= cutoff
        df = df[date_mask].copy()
        if df.empty:
            continue

        snapshot_date = int(df["trade_date"].max())
        df = df[df["trade_date"] == snapshot_date].copy()
        candidates.append((snapshot_date, path, df))

    if not candidates:
        relation = "before" if strict_before_date else "on or before"
        raise FileNotFoundError(
            f"No non-empty {index_code} weight snapshot found {relation} {cutoff}"
        )

    snapshot_date, path, df = max(candidates, key=lambda item: (item[0], item[1].name))
    df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    snapshot = WeightSnapshot(
        path=path,
        trade_date=snapshot_date,
        index_code=index_code,
        row_count=int(df.shape[0]),
    )
    return df, snapshot


def read_st_codes(data_dir: str | Path, as_of_date: int | str) -> set[str]:
    st_dir = Path(data_dir) / "stock_st"
    if not st_dir.exists():
        return set()

    cutoff = int(as_of_date)
    dates = sorted(date for date in _csv_dates(st_dir) if date <= cutoff)
    if not dates:
        return set()

    relevant_dates = dates[-2:]
    codes: set[str] = set()
    for date in relevant_dates:
        path = st_dir / f"{date}.csv"
        df = pd.read_csv(path, dtype={"ts_code": str})
        if "ts_code" in df.columns:
            codes.update(df["ts_code"].dropna().astype(str))
    return codes


def _csv_dates(directory: Path) -> list[int]:
    if not directory.exists():
        return []
    dates: list[int] = []
    for path in directory.glob("*.csv"):
        stem = path.stem
        if stem.isdigit() and len(stem) == 8:
            dates.append(int(stem))
    return dates


def _require_columns(df: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
