# Deeplearning-quantitative-trading

基于深度学习的 A 股趋势预测与模拟交易，用于"深度学习基础"课程大作业。

完整链路：股票池 → 特征工程 → TSN+GRU 训练 → 因子 IC 检验 → 回测（含手续费+GMV 组合优化） → 每日模拟交易建议。

---

## 整体设计思路

### 核心定义

- **输入**：过去 20 个交易日的 49 维原始特征（量价/技术/基本面/资金流）
- **输出**：未来 1 日超额收益（相对沪深 300）的预测分数
- **建模**：Learning to Rank — MSE + Pairwise Rank Loss
- **策略**：低换手 Top-N 缓冲选股 + GMV 权重优化 + 目标波动率控制

### 设计原则

1. **严禁未来信息泄露**：特征只到 T-1、逐日截面标准化、权重快照取 as_of_date 前
2. **数据与模型同等重要**：15 个技术指标 + 5 段 GRU 时序建模
3. **回测接近现实**：双向佣金 + 卖出印花税 + GMV 优化 + 行业/波动率风控

### 时间划分

数据范围：2016-01-04 ~ 最新（约 2523 个交易日）

| 时期 | 范围 | 用途 |
|------|------|------|
| ~~2016-2018~~ | — | 跳过（熔断+贸易战，市场结构差异大） |
| train | 2019-01 ~ 2024-12 | 评估模型训练（6 年） |
| valid | 2025-01 ~ 2025-12 | 评估模型验证 + 回测 |
| test | 2026-01 ~ 2026-05 | 比赛模型数据来源 |

### 双模型策略

| | 评估模型 | 比赛模型 |
|------|---------|---------|
| 训练数据 | train (2019-2024) | train+valid+test 合并 (2019-2026.5) |
| 早停验证 | valid (2025) | 合并后最后 10% |
| 用途 | 报告回测展示 | 6/1 起每日预测 |
| 命令 | `train_tsn.py` | `train_tsn.py --include-valid` |
| 输出 | `tsn_eval_best.pt` | `tsn_full_best.pt` |

---

## 项目结构

```text
src/
  stock_pool.py       # 动态股票池构建（沪深300增强）
  preprocess.py       # 特征计算（技术指标、标签构造、标准化）
  preprocess_seq.py   # 时序特征预处理 → TSN 序列 npz
  modeling.py         # TSN / TSN+GRU / Linear 模型 + 训练循环 + 评估
  backtest.py         # 回测引擎（手续费 + GMV + 风控）
  portfolio.py        # GMV 优化 + 目标波动率控制

scripts/
  build_stock_pool.py       # 生成指定日期股票池
  preprocess_sequences.py   # 生成序列 npz（一次预处理、三种用途）
  train_tsn.py              # 训练评估/比赛模型（--include-valid 切换）
  run_backtest.py           # 历史回测
  factor_analysis.py        # 单因子 Rank IC 检验 + 相关性矩阵
  predict_latest.py         # 每日盘后买卖建议

tests/
  test_stock_pool.py
  test_sequence_preprocess.py
  test_modeling.py
  test_backtest.py
```

---

## 环境

```bash
pip install -r requirements.txt
```

依赖：`pandas>=2.0`, `pyarrow>=14.0`, `torch>=2.0`, `scipy`（GMV 优化）

---

## 1. 股票池

沪深 300 增强，逻辑在 `src/stock_pool.py`。

- 基准 `000300.SH`，动态取 as_of_date 前最新权重快照
- 剔除北交所、ST 股、无行情、vol=0、成交额 < 1000 万
- 约 250-300 只，流动性好、聚焦可交易标的

---

## 2. 特征工程

### 特征体系（49 raw → 196 维）

| 类别 | 数量 | 内容 |
|------|------|------|
| 量价 | 18 | 收益率/振幅/跳空/收盘位置/vwap偏离/量价变化/动量(5/10/20)/波动率(5/10/20)/均线偏离(5/10/20)/成交额均值(5/10/20) |
| 技术指标 | **15** | MACD(3) / RSI(6,14) / **KDJ(3)** / 布林带(2) / **ATR** / OBV变化 / MFI / **换手率变化** / 行业相对收益 |
| 基本面 | 8 | 换手率/量比/PE/PB/PS(对数化)/股息率/总市值/流通市值 |
| 资金流向 | 8 | 主力/超大单净流入占比 + 5/10 日均值 |

每个 raw 衍生 `_z`(z-score) + `_rank`(截面百分位) + `_missing`(缺失标记) → 49×4=**196 维**。

### 技术指标详情

| 指标 | 参数 | 逻辑 |
|------|------|------|
| MACD | (12,26,9) | DIF/DEA/柱 |
| RSI | (6, 14) | 短+中周期超买超卖 |
| KDJ | (9,3,3) | 短期反转信号 |
| 布林带 | (20, 2σ) | 波动率通道，宽度+%B |
| ATR | (14) | 归一化真实波幅 |
| OBV 变化率 | — | 量价背离 |
| MFI | (14) | 量价加权 RSI |
| 换手率变化率 | (5) | 异常活跃度 |
| 行业相对收益 | (5d) | 同行业内 5 日收益排名 |

### 防泄露约束

- 特征仅用 T-1 及之前 20 天
- 标准化在每日截面内进行（非全局）
- 训练/验证/测试按时间划分，禁止随机打乱
- 股票池权重严格取 as_of_date 前最新快照

---

## 3. 模型

### 架构：TSN+GRU（主模型）

```
输入 [batch, 20天, 196维]
  → 切 5 段 [batch*5, 4天, 196]
  → Conv1d(k3) → BN → GELU → Dropout(0.3) → AdaptiveAvgPool1d
  → Reshape [batch, 5段, 128]
  → GRU(128→128, 单向)
  → Attention 聚合
  → MLP [128 → 128 → 1]
```

### 训练配置

| 参数 | 值 | 说明 |
|------|------|------|
| hidden_dim | 128 | 隐藏维度 |
| num_segments | 5 | 每段 4 天，GRU 5 步 |
| dropout | 0.3 | 防过拟合 |
| rank_weight (λ) | 0.5 | 排序损失权重 |
| 优化器 | AdamW(lr=1e-3, wd=1e-4) | — |
| 调度器 | ReduceLROnPlateau(×0.5, p=3) | — |
| 梯度裁剪 | max_norm=1.0 | — |
| Early Stop | patience=8, 按 ICIR | — |
| Epochs | 50 max | — |

### 损失函数

```
Loss = MSE(pred, y) + 0.5 × PairwiseRankLoss(pred, y)
```
PairwiseRankLoss：同日截面内股票对的排序一致性（softplus）

---

## 4. 因子检验

`scripts/factor_analysis.py` 对验证集做单因子 Rank IC 检验，输出：

| 输出 | 文件 |
|------|------|
| 因子 IC 排序表 | `outputs/factor_analysis/factor_ic_valid.csv` |
| Top-20 因子相关性矩阵 | `outputs/factor_analysis/factor_correlation.csv` |

报告中可用 IC 表证明特征有效性，用相关性热力图证明特征信息源多样。

---

## 5. 交易策略与风控

### 选股：低换手 Top-N 缓冲

| 参数 | 默认值 |
|------|--------|
| n_holdings | 10 |
| max_sell | 2 |
| buffer_rank | 30 |

规则：持仓排名 > 30 → 卖出候选；每日最多卖 2 只、买回补齐；满仓、不加杠杆。

### 权重：GMV 优化（可选，--weight-method gmv）

$$ \min_w \left( w^T\Sigma w + \lambda \sum |w_i - w_{i,prev}| \right) $$

- 单股权重 [5%, 35%]，∑w=1
- 换手惩罚 λ=0.01
- SLSQP 求解，失败退回等权

### 降波：目标波动率控制

预期年化波动 > 12% → 等比缩放，余量转现金

### 风控

- 单行业 ≤ 3 只
- 20 日波动率 > 8% → 不优先买入
- 已持仓高波动 → 进入卖出候选

---

## 6. 回测

### 成本（默认开启）

- 佣金：0.025%（买卖双向）
- 印花税：0.1%（卖出单向）

### 指标

年化收益 / 夏普比率 / 最大回撤 / 胜率 / 盈亏比 / 基准对比 / 换手率 / 总成本

### 示例结果（评估模型，2025 年验证期）

| 指标 | 策略 | 沪深 300 |
|------|------|---------|
| 年化收益 | 27.57% | 23.49% |
| 夏普比率 | 1.98 | — |
| 最大回撤 | -7.06% | — |
| 胜率 | 53.9% | — |
| 总成本占比 | 0.93% | — |

---

## 7. 模拟交易（6/1-6/12）

### 每日流程

```
盘后 18:00:
  python scripts/predict_latest.py --checkpoint outputs/models/tsn_full_best.pt

次日 9:30-15:00:
  同花顺 → 模拟大赛 → "深度学习基础-2026" → 按清单下单
```

---

## 8. 完整运行

```bash
# 全流程（推荐直接运行 run_all.bat）
python scripts/build_stock_pool.py --date 20260527 --output outputs/stock_pool_20260527.csv
python scripts/preprocess_sequences.py                                    # 一次预处理
python scripts/train_tsn.py --model-type tsn_gru                           # 评估模型
python scripts/train_tsn.py --model-type tsn_gru --include-valid           # 比赛模型
python scripts/run_backtest.py --predictions outputs/predictions/tsn_eval_valid_predictions.csv --output-dir outputs/backtest/tsn_eval_valid
python scripts/factor_analysis.py --split valid                            # 因子检验

# 每日模拟交易
python scripts/predict_latest.py --checkpoint outputs/models/tsn_full_best.pt
```

---

## 9. 报告可写亮点

- **防泄露设计**：三重机制 + 测试验证
- **15 个技术指标**：MACD/RSI/KDJ/布林带/ATR/OBV/MFI + 行业相对收益
- **因子 IC 检验**：单因子 Rank IC 表 + 相关性矩阵，证明特征有效性
- **TSN+GRU**：5 段分割 + GRU 保留时间顺序，消融可验证
- **GMV 组合优化**：带换手惩罚 + 目标波动率控制，夏普 1.98/回撤 -7%
- **双模型**：评估模型报告 + 比赛模型吃满全部数据
- **含手续费回测**：双向佣金 + 印花税

---

## 10. 测试

```bash
python -m unittest tests.test_modeling -v   # 3 tests (快速，无需数据)
python -m unittest tests.test_stock_pool -v  # 4 tests
python -m unittest tests.test_backtest -v    # 5 tests
```
