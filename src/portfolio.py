import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path


def optimize_gmv_with_penalty(
    cov_matrix: pd.DataFrame,
    prev_weights: pd.Series,
    penalty_lambda: float = 0.01,
    min_weight: float = 0.05,
    max_weight: float = 0.35,
) -> pd.Series:
    """
    步聚二：带有换手率惩罚与约束的全局最小方差模型 (Constrained GMV Optimization with Turnover Penalty)
    
    参数:
        cov_matrix: 股票收益率协方差矩阵 (N x N DataFrame)
        prev_weights: 前一期各个股票的权重 (如果不持有则为0)
        penalty_lambda: 换手率惩罚系数
        min_weight: 投资组合中单只资产权重下限 (强制分散化)
        max_weight: 投资组合中单只资产权重上限
        
    返回:
        pd.Series: 各资产的最优权重
    """
    assets = cov_matrix.columns
    n = len(assets)
    if n == 0:
        return pd.Series(dtype=float)
        
    # 如果候选股票数不足以满足下限，放宽下限
    actual_min = min(min_weight, 1.0 / n) if n > 0 else min_weight
    actual_max = max(max_weight, 1.0 / n) if n > 0 else max_weight

    cov_np = cov_matrix.values
    prev_w_np = prev_weights.reindex(assets).fillna(0.0).infer_objects(copy=False).values
    
    # 初始猜测值为等权
    w0 = np.ones(n) / n
    
    # 目标函数： w^T * Sigma * w + lambda * sum(|w_i - w_{i, prev}|)
    def objective(w):
        var = w.T @ cov_np @ w
        penalty = penalty_lambda * np.sum(np.abs(w - prev_w_np))
        return var + penalty
        
    # 边界与等式约束：全额投资权重和为 1
    bounds = [(actual_min, actual_max) for _ in range(n)]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    
    res = minimize(
        objective, 
        w0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    # SLSQP 未必100%收敛，若失败则退市等权
    final_w = res.x if res.success else w0
        
    return pd.Series(final_w, index=assets)


def apply_volatility_control(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
    target_vol: float = 0.12,
    annualization_factor: float = 252.0
) -> tuple[pd.Series, float]:
    """
    步骤三：目标波动率控制 (Target Volatility Control)
    
    如果预期年化波动率高于阈值，则将风险资产权重进行降波缩放，将剩余部分转为无风险现金部位。
    
    参数:
        weights: 优化器给出的初始风险资产权重
        cov_matrix: 股票收益率协方差矩阵 (日频)
        target_vol: 年化波动率控制上限 (默认 12%)
        annualization_factor: 日频转年化系数
        
    返回:
        (调整后的资产权重 Series, 现金权重 float)
    """
    if weights.empty:
        return weights, 1.0
        
    w_np = weights.values
    cov_np = cov_matrix.loc[weights.index, weights.index].values
    
    daily_var = w_np.T @ cov_np @ w_np
    ann_vol = np.sqrt(daily_var * annualization_factor)
    
    if ann_vol > target_vol:
        scale = target_vol / ann_vol
        adj_weights = weights * scale
        cash_weight = 1.0 - adj_weights.sum()
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
    """
    通用函数，利用近 N 日的行情计算所选股票的协方差矩阵
    """
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
    
    # 填充没有数据的股票，避免计算出错
    for code in codes:
        if code not in returns.columns:
            returns[code] = 0.0
            
    cov_matrix = returns[codes].cov().fillna(0.0)
    return cov_matrix
