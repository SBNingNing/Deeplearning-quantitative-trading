# Deeplearning-quantitative-trading

本项目用于完成"基于深度学习的股票趋势预测与模拟交易"大作业，实现从股票池选择、数据预处理、TSN+GRU 模型训练、预测评估、历史回测（含手续费）、交易策略、风控到每日模拟交易建议的完整链路。

---

## 整体设计思路

### 核心问题定义

- **输入**：过去 20 个交易日的量价 + 技术指标 + 基本面 + 资金流向特征序列
- **输出**：未来 1 日超额收益（相对沪深 300）的预测分数
- **建模方式**：排序学习（Learning to Rank），MSE + Pairwise Rank Loss 联合优化
- **策略**：低换手 Top-N 排名缓冲策略，每日最多换 2 只，持有 10 只

### 设计原则

1. **严禁未来信息泄露**：特征只用 T-1 及以前；标准化逐日截面进行；股票池权重严格选 as_of_date 前最新快照
2. **数据和模型同样重要**：技术指标增强 + TSN+GRU 时序建模
3. **回测尽量真实**：含双边佣金 + 卖出印花税；行业集中度 + 波动率风控

### 数据集时间划分

数据范围：2016-01-04 ~ 最新（约 2523 个交易日）

| 时期 | 时间范围 | 用途 | 选择理由 |
|------|---------|------|---------|
| ~~2016-2018~~ | — | 不使用 | 熔断后异常波动 + 贸易战单边熊市，市场结构与当前差异大 |
| 训练集 | **2019-01 ~ 2024-12** | 模型训练 | 2019 年起注册制改革+科创板，6 年覆盖多轮牛熊 |
| 验证集 | **2025-01 ~ 2025-12** | 超参选择 / early stopping | 最近完整年份，独立于训练 |
| 测试集 | **2026-01 ~ 最新** | 模拟交易准备 | 当前市场环境，最接近实盘 |

> 2019 年是 A 股市场结构分水岭（科创板开板、注册制推行），此后市场微观结构更接近当前状态。跳过 2016-2018 可节省约 30% 预处理时间，且避免了过时市场模式的干扰。

---

## 项目结构

```text
src/
  stock_pool.py       # 动态股票池构建（沪深300增强）
  preprocess.py       # 特征计算函数（技术指标、标签构造、标准化等）
  preprocess_seq.py   # 时序特征预处理，供 TSN/TSN+GRU 使用
  modeling.py         # TSN、TSN+GRU、Linear 模型 + 训练循环 + 评估
  backtest.py         # 历史回测（含手续费）、交易策略、风控

scripts/
  build_stock_pool.py       # 生成指定日期股票池
  preprocess_sequences.py   # 生成 TSN 序列 npz
  train_tsn.py              # 训练 TSN / TSN+GRU 主模型
  run_backtest.py           # 用预测分数做历史回测
  predict_latest.py         # 每日盘后生成买卖建议

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

依赖：`pandas>=2.0`, `pyarrow>=14.0`, `torch>=2.0`

训练脚本中设置了 `KMP_DUPLICATE_LIB_OK=TRUE`，用于绕过 Windows/PyTorch 环境中 OpenMP runtime 重复初始化问题。

---

## 1. 股票池选择

沪深 300 指数增强股票池，逻辑在 `src/stock_pool.py`。

核心规则：
- 基准指数 `000300.SH`，动态取 as_of_date 前最新权重快照（避免未来成分泄露）
- 剔除：北交所、ST 股、无有效行情、成交量=0、成交额 < 1000 万元
- 约 250-300 只候选股票

选择沪深 300 的原因：流动性好、降低训练和回测复杂度、聚焦可交易标的。未来可扩展到中证 800。

```bash
python scripts/build_stock_pool.py --date 20260527 --output outputs/stock_pool_20260527.csv
```

---

## 2. 数据预处理与特征工程

时序特征预处理逻辑在 `src/preprocess_seq.py`，特征计算函数在 `src/preprocess.py`。

输出 `X` 形状为 `[样本数, 20, 特征数]`。

### 特征体系（共 44 个原始特征）

| 类别 | 数量 | 示例 |
|------|------|------|
| 量价特征 | 18 | `ret_1d`, `振幅`, `跳空`, `收盘位置`, `vwap偏离`, `vol_chg`, `动量(5/10/20)`, `波动率(5/10/20)`, `均线偏离(5/10/20)`, `成交额均值(5/10/20)` |
| **技术指标** | **10** | `MACD(3)`, `RSI(6/14)`, `布林带宽度+%B`, `OBV变化率`, `资金流量指数MFI`, `行业相对收益` |
| 基本面 | 8 | 换手率, 量比, PE/PB/PS(TTM对数化), 股息率, 总市值, 流通市值 |
| 资金流向 | 7 | 主力/超大单净流入占比, 资金流向滚动均值 |

每个原始特征衍生 3 个变体：`_z`(z-score), `_rank`(截面百分位), `_missing`(缺失标记) → **最终约 132 维特征**。

### 技术指标详解

| 指标 | 说明 |
|------|------|
| MACD (12,26,9) | 趋势跟踪，含 DIF/DEA/柱 |
| RSI (6, 14) | 超买超卖，短周期+中周期 |
| 布林带 (20, 2σ) | 波动率通道，宽度+%B位置 |
| OBV 变化率 | 能量潮动量 |
| MFI (14) | 量价加权 RSI |
| 行业相对收益 | 个股 vs 同行业 5 日收益排名（截面特征亮点） |

### 多周期标签

| 标签 | 说明 |
|------|------|
| `label_excess_1d` | 1 日超额收益（主标签，训练优化目标） |
| `label_excess_3d` | 3 日超额收益（辅助分析，验证模型排序稳定性） |
| `label_excess_5d` | 5 日超额收益（辅助分析，匹配策略持有期） |

### 防泄露约束

- `trade_date = T`，特征仅用 `T-1` 及之前 20 天数据
- 标准化在**每日截面内**进行（非全局），严格避免跨期信息
- 训练/验证/测试按时间顺序划分，禁止随机打乱

---

## 3. 模型架构

### 模型对比

| 模型 | 参数量 | 时序建模 | 用途 |
|------|--------|---------|------|
| Linear Regression | ~130 | 无 | 最简基线 |
| TSN (原版) | ~30K | 段内池化 | 消融实验（验证 GRU 价值） |
| **TSN+GRU (主模型)** | **~38K** | **段间 GRU** | **主模型** |

### TSN+GRU 架构

```
输入 [batch, 20天, F维特征]
  → 切成 4 段 [batch*4, 5天, F]
  → Conv1d(k3,p1) → BN → GELU → Dropout → AdaptiveAvgPool1d  (段内编码)
  → Reshape [batch, 4段, H维]
  → GRU(H→H, 单向)  ← 核心增强：保留段间时间顺序
  → Attention 加权聚合 (学到的段重要性)
  → MLP 头 [H → H → 1]
  → 预测分数
```

**为什么加 GRU？** TSN 原版用 AdaptiveAvgPool1d 压平段内时序 + Attention 聚合段间信息，但段间只是简单加权。GRU 按时间顺序处理 4 个段向量，显式建模了"最近段 vs 早期段"的不同预测价值。

### 损失函数

```
Loss = MSE(pred, y) + λ * PairwiseRankLoss(pred, y)
```

- λ = 0.2（rank_weight），可调
- PairwiseRankLoss：同日内股票对排序一致性惩罚（softplus 形式）

### 训练配置

- 优化器：AdamW (lr=1e-3, weight_decay=1e-4)
- 学习率调度：ReduceLROnPlateau (factor=0.5, patience=3)
- 梯度裁剪：max_norm=1.0
- Early Stopping：patience=5，按验证集 ICIR
- Batch Size：1024

---

## 4. 交易策略

### 低换手 Top-N 排名缓冲策略（Buffered Top-N）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| n_holdings | 10 | 持仓股票数 |
| max_sell | 2 | 每日最大卖出数 |
| buffer_rank | 30 | 排名缓冲区 |

规则：
1. **首日**：买入模型分数最高的 10 只，等权建仓
2. **后续每日**：
   - 持仓中排名 > 30（跌出缓冲区）→ 卖出候选
   - 每日最多卖出 2 只（控制换手）
   - 买入：未持仓中分数最高 + 风控通过的股票，补足 10 只
3. 不做空，不加杠杆，尽量满仓

### 风控规则

- **行业集中度**：单行业最多持有 3 只
- **波动率过滤**：20 日波动率 > 8% 的股票不优先买入；已持仓高波动进入卖出候选
- **风控回退**：候选不足时回退到最高分，保持满仓

---

## 5. 历史回测

### 回测指标

| 指标 | 说明 |
|------|------|
| 总收益 / 年化收益 | 绝对收益水平 |
| 夏普比率 | 风险调整后收益 |
| 最大回撤 | 最大峰谷跌幅 |
| 胜率 | 日收益 > 0 的交易日占比 |
| 盈亏比 | 平均正收益 / 平均负收益 |
| 基准对比 | vs 沪深 300 指数 |
| 总换手率 / 总成本 | 交易成本统计 |

### 交易成本（默认开启）

- 佣金：0.025%（万 2.5，买卖双向）
- 印花税：0.1%（千 1，卖出单向）
- 可通过 `--commission-rate` 和 `--stamp-tax-rate` 调整

### 对比基准

报告中建议展示以下对比：
- TSN+GRU vs TSN（GRU 消融）
- TSN+GRU vs TSN（GRU 消融验证时序建模价值）
- TSN+GRU vs 沪深 300 指数（选股 Alpha）
- 有技术指标 vs 无技术指标（特征消融）

---

## 6. 模拟交易流程（6 月 1 日 - 12 日）

### 每日操作

```
盘后 (~18:00)：
1. 从科大云盘同步最新数据
2. python scripts/predict_latest.py
3. 查看建议买卖清单
4. 记录到 outputs/current_holdings.csv（脚本自动维护）

次日 (9:30-15:00)：
5. 在同花顺 APP → 模拟大赛 → "深度学习基础-2026"
6. 按清单下单（注意分批次、避免冲击成本）
7. 确保每日满仓
```

### predict_latest.py 输出示例

```
======================================================================
  TRADING RECOMMENDATIONS
======================================================================

  SELL (1 stocks):
    600519.SH  score=0.001234  rank=35  industry=食品饮料  risk=high_volatility

  BUY (1 stocks):
    000858.SZ  score=0.002567  rank=1   industry=食品饮料  risk=pass

  FINAL HOLDINGS (10 stocks):
     1. 000858.SZ  score=0.002567  rank=1   industry=食品饮料
     2. 600036.SH  score=0.002100  rank=2   industry=银行
     ...
======================================================================
```

---

## 7. 完整运行顺序

```bash
# 首次：预处理 + 训练 + 回测
python scripts/build_stock_pool.py --date 20260527 --output outputs/stock_pool_20260527.csv
python scripts/preprocess_sequences.py
python scripts/train_tsn.py --data-dir outputs/preprocessed_seq --model-type tsn_gru
python scripts/run_backtest.py --predictions outputs/predictions/tsn_valid_predictions.csv --output-dir outputs/backtest/tsn_valid

# 每日模拟交易（6月1日起）
python scripts/predict_latest.py
```

或直接运行 `run_all.bat`（预处理 → 训练 → 回测全流程）。

---

## 8. 实验报告大纲

```
1. 引言与问题定义
2. 数据与预处理
   2.1 数据源与时间划分（含为什么跳过 2016-2018 的分析）
   2.2 股票池构建
   2.3 特征工程（量价 + 技术指标 + 基本面 + 资金流向 + 行业相对特征）
   2.4 多周期标签设计
   2.5 标准化策略与防泄露设计
3. 模型设计
   3.1 排序学习建模
   3.2 TSN+GRU 架构详述
   3.3 基线模型（Linear, TSN 原版）
   3.4 训练配置
4. 实验结果与分析
   4.1 训练收敛情况
   4.2 IC / ICIR / 方向胜率
   4.3 回测资金曲线与指标对比
   4.4 消融实验（GRU / 技术指标 / 多周期标签）
   4.5 参数敏感性分析
5. 模拟交易
   5.1 操作流程
   5.2 每日调仓记录
   5.3 收益曲线（含截图）
   5.4 预实盘差异反思
6. 总结与反思
   6.1 实验亮点
   6.2 不足与改进
   6.3 组员分工
```

---

## 9. 报告可写亮点

- **防泄露设计**：权重快照、特征截止日、逐日截面标准化，均有测试验证
- **技术指标增强**：MACD/RSI/布林带/OBV/MFI + 行业相对收益（截面特征创新点）
- **多周期标签**：1d/3d/5d 超额收益，匹配策略持有期
- **TSN+GRU 时序建模**：在 TSN 基础上增加 GRU 保留段间时间顺序，消融实验可验证价值
- **含手续费的现实回测**：万 2.5 佣金 + 千 1 印花税
- **低换手缓冲策略**：减少噪声交易，适合短期模拟
- **风控层**：行业集中度 + 波动率过滤 + 回退机制

---

## 10. 测试

```bash
python -m unittest tests.test_modeling -v   # 3 tests (快速)
python -m unittest tests.test_stock_pool -v  # 4 tests (需数据)
python -m unittest tests.test_backtest -v    # 5 tests (需数据)
# 全部测试: python -m unittest discover -s tests
```
