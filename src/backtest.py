from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import available_daily_dates, read_market_index
from stock_pool import DEFAULT_INDEX_CODE


UNKNOWN_INDUSTRY = "未知"


@dataclass(frozen=True)
class BacktestMetrics:
    start_date: int
    end_date: int
    trading_days: int
    total_return: float
    annualized_return: float
    sharpe: float
    max_drawdown: float
    benchmark_total_return: float
    win_rate: float
    profit_loss_ratio: float
    total_turnover: float
    total_cost: float


def run_score_backtest(
    predictions: pd.DataFrame,
    data_dir: str | Path = "A股数据",
    index_code: str = DEFAULT_INDEX_CODE,
    initial_cash: float = 1_000_000.0,
    n_holdings: int = 10,
    rebalance_count: int | None = None,
    strategy: str = "buffered_topn",
    max_sell: int = 2,
    buffer_rank: int = 30,
    enable_risk_control: bool = True,
    max_industry_count: int = 3,
    volatility_window: int = 20,
    max_daily_volatility: float = 0.08,
    commission_rate: float = 0.00025,
    stamp_tax_rate: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, BacktestMetrics]:
    required = {"trade_date", "ts_code", "score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing required columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("Predictions are empty")
    if strategy not in {"simple_topk", "buffered_topn"}:
        raise ValueError("strategy must be either 'simple_topk' or 'buffered_topn'")
    if rebalance_count is not None:
        max_sell = rebalance_count
    if n_holdings <= 0:
        raise ValueError("n_holdings must be positive")
    if max_sell <= 0:
        raise ValueError("max_sell must be positive")
    if buffer_rank < n_holdings:
        raise ValueError("buffer_rank must be greater than or equal to n_holdings")
    if max_industry_count <= 0:
        raise ValueError("max_industry_count must be positive")
    if volatility_window <= 1:
        raise ValueError("volatility_window must be greater than 1")
    if max_daily_volatility <= 0:
        raise ValueError("max_daily_volatility must be positive")

    root = Path(data_dir)
    trading_dates = available_daily_dates(root)
    date_to_index = {date: index for index, date in enumerate(trading_dates)}
    date_to_next = {
        trading_dates[index]: trading_dates[index + 1]
        for index in range(len(trading_dates) - 1)
    }
    market = read_market_index(root, index_code).set_index("trade_date")["close"]
    industry_map = read_industry_map(root)

    scores = predictions.copy()
    scores["trade_date"] = scores["trade_date"].astype(int)
    scores["ts_code"] = scores["ts_code"].astype(str)
    scores = scores[scores["trade_date"].isin(date_to_next)].copy()
    if scores.empty:
        raise ValueError("No prediction dates have a next trading day")

    holdings: list[str] = []
    equity = initial_cash
    total_turnover = 0.0
    total_cost_amount = 0.0
    curve_rows: list[dict[str, float | int]] = []
    holding_rows: list[dict[str, str | int | float]] = []
    trade_rows: list[dict[str, str | int | float]] = []

    for trade_date, day_scores in scores.groupby("trade_date", sort=True):
        day_scores = prepare_daily_scores(day_scores)
        day_scores = apply_risk_annotations(
            day_scores=day_scores,
            data_dir=root,
            trade_date=int(trade_date),
            trading_dates=trading_dates,
            date_to_index=date_to_index,
            industry_map=industry_map,
            enable_risk_control=enable_risk_control,
            volatility_window=volatility_window,
            max_daily_volatility=max_daily_volatility,
        )
        n_sells = 0
        n_buys = 0
        if not holdings:
            new_holdings = select_buy_candidates(
                day_scores=day_scores,
                existing_holdings=[],
                excluded_codes=set(),
                slots=n_holdings,
                max_industry_count=max_industry_count,
                enable_risk_control=enable_risk_control,
            )
            for code in new_holdings:
                row = day_scores[day_scores["ts_code"] == code].iloc[0]
                trade_rows.append(
                    trade_record(int(trade_date), "buy", code, "initial", row)
                )
            holdings = new_holdings
            n_buys = len(new_holdings)
        else:
            holdings, sells, buys = rebalance_holdings(
                holdings=holdings,
                day_scores=day_scores,
                n_holdings=n_holdings,
                max_sell=max_sell,
                buffer_rank=buffer_rank,
                strategy=strategy,
                max_industry_count=max_industry_count,
                enable_risk_control=enable_risk_control,
            )
            n_sells = len(sells)
            n_buys = len(buys)
            daily_by_code = day_scores.set_index("ts_code", drop=False)
            for code in sells:
                row = daily_by_code.loc[code] if code in daily_by_code.index else None
                trade_rows.append(
                    trade_record(int(trade_date), "sell", code, strategy, row)
                )
            for code in buys:
                row = daily_by_code.loc[code]
                trade_rows.append(
                    trade_record(int(trade_date), "buy", code, strategy, row)
                )

        next_date = date_to_next[int(trade_date)]
        returns = read_forward_returns(root, int(trade_date), next_date, holdings)
        portfolio_return = float(returns["return_1d"].mean()) if not returns.empty else 0.0

        # Transaction cost: commission on both sides, stamp tax on sells only
        cost_rate = 0.0
        if n_sells > 0:
            cost_rate += (n_sells / max(n_holdings, 1)) * (commission_rate + stamp_tax_rate)
        if n_buys > 0:
            cost_rate += (n_buys / max(n_holdings, 1)) * commission_rate
        total_turnover += (n_sells + n_buys) / max(n_holdings, 1)
        total_cost_amount += cost_rate * equity
        net_return = portfolio_return - cost_rate
        equity *= 1.0 + net_return

        if int(trade_date) in market.index and next_date in market.index:
            benchmark_return = float(market.loc[next_date] / market.loc[int(trade_date)] - 1.0)
        else:
            benchmark_return = np.nan
        curve_rows.append(
            {
                "trade_date": int(trade_date),
                "next_date": int(next_date),
                "portfolio_return": portfolio_return,
                "cost_rate": cost_rate,
                "net_return": net_return,
                "benchmark_return": benchmark_return,
                "equity": equity,
            }
        )
        equal_weight = 1.0 / max(len(holdings), 1)
        daily_by_code = day_scores.set_index("ts_code", drop=False)
        for code in holdings:
            row = daily_by_code.loc[code] if code in daily_by_code.index else None
            holding_rows.append(
                {
                    "trade_date": int(trade_date),
                    "ts_code": code,
                    "weight": equal_weight,
                    "industry": row["industry"] if row is not None else UNKNOWN_INDUSTRY,
                    "volatility": float(row["volatility"]) if row is not None and pd.notna(row["volatility"]) else np.nan,
                }
            )

    curve = pd.DataFrame(curve_rows)
    holdings_frame = pd.DataFrame(holding_rows)
    trades = pd.DataFrame(trade_rows)
    metrics = compute_backtest_metrics(
        curve, initial_cash,
        total_turnover=total_turnover,
        total_cost=total_cost_amount,
    )
    return curve, holdings_frame, trades, metrics


def prepare_daily_scores(day_scores: pd.DataFrame) -> pd.DataFrame:
    df = day_scores.sort_values("score", ascending=False).drop_duplicates("ts_code").copy()
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def rebalance_holdings(
    holdings: list[str],
    day_scores: pd.DataFrame,
    n_holdings: int,
    max_sell: int,
    buffer_rank: int,
    strategy: str,
    max_industry_count: int,
    enable_risk_control: bool,
) -> tuple[list[str], list[str], list[str]]:
    rank_map = day_scores.set_index("ts_code")["rank"].to_dict()
    if strategy == "simple_topk":
        scored_holdings = sorted(
            holdings,
            key=lambda code: rank_map.get(code, np.inf),
            reverse=True,
        )
        sells = scored_holdings[: min(max_sell, len(scored_holdings))]
    else:
        sell_candidates = [
            code
            for code in holdings
            if rank_map.get(code, np.inf) > buffer_rank
        ]
        if enable_risk_control and "risk_pass" in day_scores.columns:
            risk_map = day_scores.set_index("ts_code")["risk_pass"].to_dict()
            sell_candidates.extend(
                code
                for code in holdings
                if code in risk_map and not bool(risk_map[code])
            )
            sell_candidates = list(dict.fromkeys(sell_candidates))
        sells = sorted(
            sell_candidates,
            key=lambda code: rank_map.get(code, np.inf),
            reverse=True,
        )[:max_sell]

    sell_set = set(sells)
    keep = [code for code in holdings if code not in sell_set]
    buys = select_buy_candidates(
        day_scores=day_scores,
        existing_holdings=keep,
        excluded_codes=sell_set,
        slots=max(0, n_holdings - len(keep)),
        max_industry_count=max_industry_count,
        enable_risk_control=enable_risk_control,
    )
    return (keep + buys)[:n_holdings], sells, buys


def select_buy_candidates(
    day_scores: pd.DataFrame,
    existing_holdings: list[str],
    excluded_codes: set[str],
    slots: int,
    max_industry_count: int,
    enable_risk_control: bool,
) -> list[str]:
    if slots <= 0:
        return []
    existing_set = set(existing_holdings)
    industry_counts: dict[str, int] = {}
    if enable_risk_control and "industry" in day_scores.columns:
        industry_map = day_scores.set_index("ts_code")["industry"].to_dict()
        for code in existing_holdings:
            industry = str(industry_map.get(code, UNKNOWN_INDUSTRY))
            industry_counts[industry] = industry_counts.get(industry, 0) + 1

    buys: list[str] = []
    fallback: list[str] = []
    for _, row in day_scores.iterrows():
        code = str(row["ts_code"])
        if code in existing_set or code in excluded_codes:
            continue
        if enable_risk_control and "risk_pass" in day_scores.columns and not bool(row["risk_pass"]):
            fallback.append(code)
            continue
        industry = str(row["industry"]) if "industry" in row else UNKNOWN_INDUSTRY
        if enable_risk_control and industry_counts.get(industry, 0) >= max_industry_count:
            fallback.append(code)
            continue
        buys.append(code)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(buys) == slots:
            return buys

    # Keep the strategy fully invested when risk filters are too strict.
    for code in fallback:
        if code not in buys:
            buys.append(code)
        if len(buys) == slots:
            break
    return buys


def apply_risk_annotations(
    day_scores: pd.DataFrame,
    data_dir: Path,
    trade_date: int,
    trading_dates: list[int],
    date_to_index: dict[int, int],
    industry_map: dict[str, str],
    enable_risk_control: bool,
    volatility_window: int,
    max_daily_volatility: float,
) -> pd.DataFrame:
    df = day_scores.copy()
    df["industry"] = df["ts_code"].map(industry_map).fillna(UNKNOWN_INDUSTRY)
    df["volatility"] = np.nan
    df["risk_pass"] = True
    df["risk_reason"] = "pass"
    if not enable_risk_control:
        return df

    volatility = compute_realized_volatility(
        data_dir=data_dir,
        trading_dates=trading_dates,
        date_to_index=date_to_index,
        trade_date=trade_date,
        codes=df["ts_code"].astype(str).tolist(),
        window=volatility_window,
    )
    df["volatility"] = df["ts_code"].map(volatility)
    high_volatility = df["volatility"].notna() & (df["volatility"] > max_daily_volatility)
    df.loc[high_volatility, "risk_pass"] = False
    df.loc[high_volatility, "risk_reason"] = "high_volatility"
    return df


def compute_realized_volatility(
    data_dir: Path,
    trading_dates: list[int],
    date_to_index: dict[int, int],
    trade_date: int,
    codes: list[str],
    window: int,
) -> dict[str, float]:
    if trade_date not in date_to_index:
        return {}
    end_index = date_to_index[trade_date]
    start_index = max(0, end_index - window - 1)
    history_dates = trading_dates[start_index:end_index]
    if len(history_dates) < 3:
        return {}

    code_set = set(codes)
    frames: list[pd.DataFrame] = []
    for date in history_dates:
        path = data_dir / "daily" / f"{date}.csv"
        day = pd.read_csv(path, usecols=["ts_code", "close"], dtype={"ts_code": str})
        day = day[day["ts_code"].isin(code_set)].copy()
        day["trade_date"] = date
        frames.append(day)
    if not frames:
        return {}
    history = pd.concat(frames, ignore_index=True).sort_values(["ts_code", "trade_date"])
    history["close"] = pd.to_numeric(history["close"], errors="coerce")
    history["ret"] = history.groupby("ts_code")["close"].pct_change()
    volatility = history.groupby("ts_code")["ret"].std(ddof=0).dropna()
    return {str(code): float(value) for code, value in volatility.items()}


def read_industry_map(data_dir: Path) -> dict[str, str]:
    path = data_dir / "basic.csv"
    basic = pd.read_csv(path, usecols=["ts_code", "industry"], dtype={"ts_code": str})
    basic["industry"] = basic["industry"].fillna(UNKNOWN_INDUSTRY).astype(str)
    return dict(zip(basic["ts_code"], basic["industry"]))


def trade_record(
    trade_date: int,
    action: str,
    code: str,
    reason: str,
    score_row: pd.Series | None,
) -> dict[str, str | int | float]:
    record: dict[str, str | int | float] = {
        "trade_date": trade_date,
        "action": action,
        "ts_code": code,
        "reason": reason,
    }
    if score_row is not None:
        record["score"] = float(score_row["score"])
        record["rank"] = int(score_row["rank"])
        record["industry"] = str(score_row.get("industry", UNKNOWN_INDUSTRY))
        record["volatility"] = (
            float(score_row["volatility"])
            if "volatility" in score_row and pd.notna(score_row["volatility"])
            else np.nan
        )
        record["risk_reason"] = str(score_row.get("risk_reason", "pass"))
    else:
        record["score"] = np.nan
        record["rank"] = np.nan
        record["industry"] = UNKNOWN_INDUSTRY
        record["volatility"] = np.nan
        record["risk_reason"] = "missing_score"
    return record


def read_forward_returns(
    data_dir: Path,
    trade_date: int,
    next_date: int,
    holdings: list[str],
) -> pd.DataFrame:
    if not holdings:
        return pd.DataFrame(columns=["ts_code", "return_1d"])
    today = pd.read_csv(
        data_dir / "daily" / f"{trade_date}.csv",
        usecols=["ts_code", "close"],
        dtype={"ts_code": str},
    ).rename(columns={"close": "close_t"})
    next_day = pd.read_csv(
        data_dir / "daily" / f"{next_date}.csv",
        usecols=["ts_code", "close"],
        dtype={"ts_code": str},
    ).rename(columns={"close": "close_t1"})
    returns = today.merge(next_day, on="ts_code", how="inner")
    returns = returns[returns["ts_code"].isin(holdings)].copy()
    returns["return_1d"] = returns["close_t1"] / returns["close_t"] - 1.0
    return returns[["ts_code", "return_1d"]]


def compute_backtest_metrics(
    curve: pd.DataFrame,
    initial_cash: float,
    total_turnover: float = 0.0,
    total_cost: float = 0.0,
) -> BacktestMetrics:
    if curve.empty:
        raise ValueError("Backtest curve is empty")
    returns = curve["portfolio_return"].astype(float)
    equity = curve["equity"].astype(float)
    total_return = float(equity.iloc[-1] / initial_cash - 1.0)
    annualized_return = float((1.0 + total_return) ** (252.0 / len(curve)) - 1.0)
    return_std = float(returns.std(ddof=0))
    sharpe = float(returns.mean() / return_std * np.sqrt(252.0)) if return_std > 0 else np.nan
    running_max = equity.cummax()
    max_drawdown = float((equity / running_max - 1.0).min())
    benchmark = curve["benchmark_return"].dropna()
    if benchmark.empty:
        benchmark_total = np.nan
    else:
        benchmark_total = float((1.0 + benchmark).prod() - 1.0)
    win_rate = float((returns > 0).mean()) if len(returns) > 0 else 0.0
    pos_mean = returns[returns > 0].mean()
    neg_mean = abs(returns[returns < 0].mean())
    profit_loss_ratio = float(pos_mean / neg_mean) if neg_mean and neg_mean > 0 else float("nan")
    return BacktestMetrics(
        start_date=int(curve["trade_date"].iloc[0]),
        end_date=int(curve["trade_date"].iloc[-1]),
        trading_days=int(len(curve)),
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        benchmark_total_return=benchmark_total,
        win_rate=win_rate,
        profit_loss_ratio=profit_loss_ratio,
        total_turnover=total_turnover,
        total_cost=total_cost,
    )


def write_backtest_outputs(
    output_dir: str | Path,
    curve: pd.DataFrame,
    holdings: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: BacktestMetrics,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    curve.to_csv(output / "equity_curve.csv", index=False)
    holdings.to_csv(output / "holdings.csv", index=False)
    trades.to_csv(output / "trades.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(json_safe(asdict(metrics)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value
