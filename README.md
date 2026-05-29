# Deeplearning-quantitative-trading

本项目用于完成“基于深度学习的股票趋势预测与模拟交易”大作业，目前已经实现从股票池选择、数据预处理、TSN 模型训练、MLP baseline、预测评估、历史回测、交易策略到轻量风控的完整代码链路。

当前股票池采用 **沪深300指数增强股票池**，目标是在保证流程可靠和避免未来信息泄露的前提下，完成可复现的深度学习选股与模拟交易实验。

## 当前状态

已完成：

- 股票池选择：沪深300成分股动态股票池，剔除北交所、ST、无效行情和低成交额股票。
- 截面特征预处理：生成 MLP baseline 可用的表格特征。
- 序列特征预处理：生成 TSN 可用的 `[样本数, 20, 特征数]` 三维时序输入。
- 模型训练：TSN 主模型和 MLP baseline。
- 模型评估：loss、daily IC、ICIR、方向胜率。
- 历史回测：已经实现，可基于模型预测分数生成资金曲线、持仓、调仓和收益指标。
- 交易策略：低换手 Top-N 排名缓冲策略。
- 风控模块：行业集中度限制和历史波动率过滤。
- 测试：当前 `13` 个单元测试通过。

当前尚未完成或需要后续补充：

- 完整训练集上的正式 TSN 训练结果还需要重新跑全量数据。
- 新闻文本数据暂未使用。
- 模拟交易比赛的真实下单记录和截图需要赛后补充到报告。

## 项目结构

```text
src/
  stock_pool.py       # 动态股票池构建
  preprocess.py       # 截面特征预处理，供 MLP baseline 使用
  preprocess_seq.py   # 时序特征预处理，供 TSN 使用
  modeling.py         # TSN、MLP、训练循环和评估指标
  backtest.py         # 历史回测、交易策略和风控

scripts/
  build_stock_pool.py       # 生成指定日期股票池
  preprocess_features.py    # 生成表格特征 parquet
  preprocess_sequences.py   # 生成 TSN 序列 npz
  train_tsn.py              # 训练 TSN 主模型
  train_mlp_baseline.py     # 训练 MLP baseline
  run_backtest.py           # 用预测分数做历史回测

tests/
  test_stock_pool.py
  test_preprocess.py
  test_sequence_preprocess.py
  test_modeling.py
  test_backtest.py
```

## 环境

依赖见 `requirements.txt`：

```bash
pip install -r requirements.txt
```

主要依赖：

- `pandas`
- `pyarrow`
- `torch`

训练脚本中设置了 `KMP_DUPLICATE_LIB_OK=TRUE`，用于绕过当前 Windows/PyTorch 环境中可能出现的 OpenMP runtime 重复初始化问题。

## 1. 股票池选择

逻辑在 `src/stock_pool.py`，入口在 `scripts/build_stock_pool.py`。

核心规则：

- 基准指数固定为 `000300.SH`。
- 对任一交易日，只使用该日期之前最近的非空沪深300成分权重文件，避免未来成分泄露。
- 剔除北交所股票、ST 股票、无有效当日行情、成交量为 0 或成交额低于阈值的股票。
- 默认最低成交额为 `10000`，单位与原始数据一致，即千元。

示例：

```bash
python scripts/build_stock_pool.py --date 20260527
```

保存结果：

```bash
python scripts/build_stock_pool.py --date 20260527 --output outputs/stock_pool_20260527.csv --summary-output outputs/stock_pool_20260527.json
```

## 2. 数据预处理

### 截面特征

逻辑在 `src/preprocess.py`，用于 MLP baseline。

约束：

- 样本交易日为 `trade_date = T`。
- 特征只使用 `feature_end_date = T-1` 及以前 20 个交易日的数据。
- 标签使用 `close(T+1) / close(T) - 1`，并计算相对沪深300的 1 日超额收益。
- 股票池使用 `feature_end_date` 构建，避免使用交易日当天盘后信息。

运行：

```bash
python scripts/preprocess_features.py
```

输出：

- `outputs/preprocessed/features_train.parquet`
- `outputs/preprocessed/features_valid.parquet`
- `outputs/preprocessed/features_test.parquet`
- `outputs/preprocessed/preprocess_summary.json`

### TSN 序列特征

逻辑在 `src/preprocess_seq.py`，用于 TSN 主模型。

输出 `X` 形状为：

```text
[样本数, 20, 特征数]
```

每个样本包含：

- `X`
- `y`
- `trade_date`
- `feature_end_date`
- `label_end_date`
- `feature_dates`
- `ts_code`
- `weight`
- `industry`

运行：

```bash
python scripts/preprocess_sequences.py
```

输出：

- `outputs/preprocessed_seq/sequences_train.npz`
- `outputs/preprocessed_seq/sequences_valid.npz`
- `outputs/preprocessed_seq/sequences_test.npz`
- `outputs/preprocessed_seq/sequence_summary.json`

小样本调试：

```bash
python scripts/preprocess_sequences.py --start-date 20260105 --end-date 20260109 --output-dir outputs/preprocessed_seq_sample
```

## 3. 模型训练

### TSN 主模型

TSN 实现在 `src/modeling.py`，训练入口为 `scripts/train_tsn.py`。

模型结构：

- 输入 20 个交易日的特征序列。
- 切成 4 个时间段，每段 5 个交易日。
- 每段共享 `Conv1d -> BatchNorm -> GELU -> Dropout -> AdaptiveAvgPool` 编码器。
- 使用 attention consensus 聚合时间段特征。
- MLP 回归头输出股票预测分数。

训练：

```bash
python scripts/train_tsn.py --data-dir outputs/preprocessed_seq
```

默认输出：

- `outputs/models/tsn_best.pt`
- `outputs/models/tsn_history.csv`
- `outputs/predictions/tsn_train_predictions.csv`
- `outputs/predictions/tsn_valid_predictions.csv`
- `outputs/predictions/tsn_test_predictions.csv`
- `outputs/predictions/tsn_metrics.json`

如果 8GB 显存不足，可以降低 batch size：

```bash
python scripts/train_tsn.py --data-dir outputs/preprocessed_seq --batch-size 512
```

### MLP baseline

MLP baseline 使用截面特征，用于报告中和 TSN 对比。

训练：

```bash
python scripts/train_mlp_baseline.py --data-dir outputs/preprocessed
```

默认输出：

- `outputs/models/mlp_best.pt`
- `outputs/models/mlp_history.csv`
- `outputs/predictions/mlp_train_predictions.csv`
- `outputs/predictions/mlp_valid_predictions.csv`
- `outputs/predictions/mlp_test_predictions.csv`
- `outputs/predictions/mlp_metrics.json`

## 4. 历史回测

历史回测已经实现，逻辑在 `src/backtest.py`，入口在 `scripts/run_backtest.py`。

回测输入是模型预测文件，至少需要：

- `trade_date`
- `ts_code`
- `score`

运行 TSN 验证集回测：

```bash
python scripts/run_backtest.py --predictions outputs/predictions/tsn_valid_predictions.csv --output-dir outputs/backtest/tsn_valid
```

回测输出：

- `equity_curve.csv`：每日资金曲线、组合收益、基准收益。
- `holdings.csv`：每日持仓、权重、行业、波动率。
- `trades.csv`：每日买卖记录、模型分数、排名、风控原因。
- `metrics.json`：总收益、年化收益、夏普比率、最大回撤、基准总收益。

当前历史回测支持两个策略：

- `buffered_topn`：默认策略，低换手 Top-N 排名缓冲。
- `simple_topk`：每天卖出最低分持仓并买入最高分候选。

示例：

```bash
python scripts/run_backtest.py --predictions outputs/predictions/tsn_valid_predictions.csv --output-dir outputs/backtest/tsn_valid --strategy buffered_topn
```

## 5. 交易策略

默认采用 **低换手 Top-N 排名缓冲策略**：

- 初始买入模型分数最高的 10 只股票并等权持有。
- 后续只有当持仓排名跌出前 30 时才进入卖出候选。
- 每天最多卖出 2 只。
- 卖出后买入当前未持仓股票中排名最高的股票，补足 10 只。
- 不做空，不加杠杆，尽量满仓。

可调参数：

```bash
--n-holdings 10
--max-sell 2
--buffer-rank 30
```

选择该策略的原因：

- 股票短期收益噪声大，模型分数不宜驱动高频大换仓。
- 排名缓冲可以减少无意义交易。
- 10 只股票兼顾分散和模型信号集中度。
- 每日最多换 2 只，适合 10 个交易日左右的模拟交易周期。

## 6. 风控模块

回测默认开启轻量风控层：

- 单行业最多持有 3 只股票，降低行业集中度。
- 使用交易日前 20 个交易日收盘价计算日收益波动率。
- 波动率高于 `0.08` 的股票不优先买入。
- 已持仓股票若触发高波动，会进入卖出候选。
- 风控过严导致候选不足时，会回退到最高分候选，以尽量保持满仓。

可调参数：

```bash
--max-industry-count 3
--volatility-window 20
--max-daily-volatility 0.08
--disable-risk-control
```

## 7. 推荐完整运行顺序

首次完整实验建议按以下顺序运行：

```bash
python scripts/build_stock_pool.py --date 20260527 --output outputs/stock_pool_20260527.csv --summary-output outputs/stock_pool_20260527.json
python scripts/preprocess_features.py
python scripts/preprocess_sequences.py
python scripts/train_tsn.py --data-dir outputs/preprocessed_seq
python scripts/train_mlp_baseline.py --data-dir outputs/preprocessed
python scripts/run_backtest.py --predictions outputs/predictions/tsn_valid_predictions.csv --output-dir outputs/backtest/tsn_valid
python scripts/run_backtest.py --predictions outputs/predictions/mlp_valid_predictions.csv --output-dir outputs/backtest/mlp_valid
```

正式模拟交易前，每天盘后更新数据后，可以重新生成最新预测，再根据 `tsn_test_predictions.csv` 或最新日期预测文件中的分数进行人工下单。

## 8. 代码检查

当前已执行检查：

```bash
python -m unittest discover -s tests
```

结果：

```text
Ran 13 tests
OK
```

还执行了基于 `ast.parse` 的语法检查，结果为 `syntax_ok`。

说明：当前 Windows 工作区中直接运行 `python -m compileall src scripts tests` 会因为 `__pycache__` 写入权限报 `PermissionError`，但单元测试已经实际导入并执行核心模块，语法检查也通过；这属于字节码缓存写入权限问题，不是代码语法问题。

## 9. 当前样本验证结果

为了确认链路可运行，已经用 2026 年 1 月 5 日至 2026 年 1 月 7 日附近的小样本跑通过：

- 序列预处理。
- TSN 1 epoch 冒烟训练。
- MLP 1 epoch 冒烟训练。
- TSN 预测文件回测。
- 风控回测输出。

样本太短，只用于验证代码链路，不用于评价模型真实效果。正式报告应使用完整训练集和验证集结果。

## 10. 报告可写重点

报告中可以围绕以下点展开：

- 数据处理严格避免未来信息泄露。
- 使用沪深300增强股票池，降低全 A 股训练和回测复杂度。
- TSN 将 20 日序列切分为多个时间段，捕捉短期时序模式。
- MLP baseline 用于证明 TSN 设计不是孤立实验。
- 评价指标包含 loss、IC、ICIR、方向胜率和历史回测指标。
- 交易策略采用低换手排名缓冲，适合短期模拟交易。
- 风控层控制行业集中度和高波动个股风险。

