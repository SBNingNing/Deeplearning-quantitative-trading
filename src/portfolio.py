"""Portfolio optimization: GMV with turnover penalty + target volatility control."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path


def optimize_gmv_with_penalty(
    cov_matrix: pd.DataFrame,
    prev_weights: pd.Series,
    score_alpha: pd.Series | None = None,
    penalty_lambda: float = 0.005,
    score_tilt: float = 0.003,
    min_weight: float = 0.05,
    max_weight: float = 0.20,
) -> pd.Series:
    """Global minimum variance with turnover penalty.

    Objective:
        min  w^T Σ w  + λ Σ|w_i - w_{i,prev}|

    Parameters
    ----------
    cov_matrix : (N x N) daily return covariance
    prev_weights : previous day's weights (0 for new entries)
    penalty_lambda : turnover penalty coefficient
    min_weight / max_weight : per-asset bounds

    Returns
    -------
    pd.Series of optimal weights indexed by asset code.
    """
    assets = cov_matrix.columns.tolist()
    n = len(assets)
    if n == 0:
        return pd.Series(dtype=float)

    actual_min = min(min_weight, 1.0 / n) if n > 0 else min_weight
    actual_max = max(max_weight, 1.0 / n) if n > 0 else max_weight

    cov_np = cov_matrix.values.astype(np.float64)
    prev_w_np = prev_weights.reindex(assets).fillna(0.0).values.astype(np.float64)
    if score_alpha is None or score_alpha.empty:
        alpha_np = np.zeros(n, dtype=np.float64)
    else:
        alpha_np = score_alpha.reindex(assets).fillna(0.0).values.astype(np.float64)
    w0 = np.ones(n) / n

    def objective(w: np.ndarray) -> float:
        var = float(w.T @ cov_np @ w)
        penalty = penalty_lambda * float(np.sum(np.abs(w - prev_w_np)))
        tilt = score_tilt * float(alpha_np @ w)
        return var + penalty - tilt

    bounds = [(actual_min, actual_max) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(
        objective, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    final_w = res.x if res.success else w0
    return pd.Series(final_w, index=assets)


def portfolio_annualized_volatility(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
    annualization_factor: float = 252.0,
) -> float:
    """Compute expected annualized volatility from daily covariance."""
    if weights.empty:
        return 0.0
    aligned = cov_matrix.loc[weights.index, weights.index]
    w_np = weights.values.astype(np.float64)
    cov_np = aligned.values.astype(np.float64)
    daily_var = float(w_np.T @ cov_np @ w_np)
    return float(np.sqrt(max(daily_var, 0.0) * annualization_factor))


def apply_volatility_control(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
    target_vol: float = 0.12,
    annualization_factor: float = 252.0,
) -> tuple[pd.Series, float]:
    """Target volatility control.

    If expected annualized vol > target_vol, scale down proportionally,
    allocating remainder to cash.

    Returns (adjusted_weights, cash_weight).
    """
    if weights.empty:
        return weights, 1.0

    w_np = weights.values.astype(np.float64)
    cov_np = cov_matrix.loc[weights.index, weights.index].values.astype(np.float64)

    daily_var = float(w_np.T @ cov_np @ w_np)
    ann_vol = float(np.sqrt(max(daily_var, 0.0) * annualization_factor))

    if ann_vol > target_vol:
        scale = target_vol / ann_vol
        adj_weights = weights * scale
        cash_weight = 1.0 - float(adj_weights.sum())
        return adj_weights, cash_weight

    return weights, 0.0


def compute_covariance_cache(
    data_dir: Path,
    trading_dates: list[int],
    date_to_index: dict[int, int],
    trade_date: int,
    codes: list[str],
    window: int,
) -> pd.DataFrame:
    """Compute daily-return covariance matrix over the last `window` days."""
    if trade_date not in date_to_index or not codes:
        return pd.DataFrame()

    end_index = date_to_index[trade_date]
    start_index = max(0, end_index - window - 1)
    history_dates = trading_dates[start_index:end_index]

    if len(history_dates) < 3:
        return pd.DataFrame(0.0, index=codes, columns=codes)

    code_set = set(codes)
    frames: list[pd.DataFrame] = []

    for date in history_dates:
        path = data_dir / "daily" / f"{date}.csv"
        day = pd.read_csv(path, usecols=["ts_code", "close"], dtype={"ts_code": str})
        day = day[day["ts_code"].isin(code_set)].copy()
        day["trade_date"] = date
        frames.append(day)

    if not frames:
        return pd.DataFrame(0.0, index=codes, columns=codes)

    history = pd.concat(frames, ignore_index=True)
    history["close"] = pd.to_numeric(history["close"], errors="coerce")

    pivot_close = history.pivot(index="trade_date", columns="ts_code", values="close")
    returns = pivot_close.pct_change().dropna(how="all")

    for code in codes:
        if code not in returns.columns:
            returns[code] = 0.0

    cov_matrix = returns[codes].cov().fillna(0.0)
    return cov_matrix
