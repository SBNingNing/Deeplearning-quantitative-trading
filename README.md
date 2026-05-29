# Deeplearning-quantitative-trading

## 沪深300指数增强股票池

本仓库当前实现了大作业第一步：构建沪深300指数增强的动态股票池。逻辑在 `src/stock_pool.py`，命令行入口在 `scripts/build_stock_pool.py`。

核心规则：

- 基准指数固定为 `000300.SH`。
- 对任一交易日，只使用该日期之前最近的非空沪深300成分权重文件，避免未来成分泄露。
- 剔除北交所股票、ST 股票、无有效当日行情、成交量为 0 或成交额低于阈值的股票。
- 默认最低成交额为 `10000`，单位与原始数据一致，即千元。

常用命令：

```bash
python scripts/build_stock_pool.py --date 20260527
```

保存股票池和摘要：

```bash
python scripts/build_stock_pool.py --date 20260527 --output outputs/stock_pool_20260527.csv --summary-output outputs/stock_pool_20260527.json
```

不指定日期时，脚本会使用 `A股数据/daily/` 中最新的交易日文件。

运行测试：

```bash
python -m unittest discover -s tests
```

## 数据预处理

预处理逻辑在 `src/preprocess.py`，命令行入口在 `scripts/preprocess_features.py`。它会把按日截面的行情、基本面和资金流数据转成训练样本，并强制满足：

- 样本交易日为 `trade_date = T`。
- 特征只使用 `feature_end_date = T-1` 及以前 20 个交易日的数据。
- 标签使用 `close(T+1) / close(T) - 1`，并同时计算相对沪深300的 1 日超额收益。
- 股票池使用 `feature_end_date` 构建，避免使用交易日当天盘后信息。

生成完整训练、验证、测试集：

```bash
python scripts/preprocess_features.py
```

生成小范围样例：

```bash
python scripts/preprocess_features.py --start-date 20260105 --end-date 20260109 --output-dir outputs/preprocessed_sample
```

默认输出：

- `outputs/preprocessed/features_train.parquet`
- `outputs/preprocessed/features_valid.parquet`
- `outputs/preprocessed/features_test.parquet`
- `outputs/preprocessed/preprocess_summary.json`
