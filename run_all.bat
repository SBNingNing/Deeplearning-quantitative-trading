@echo off
chcp 65001 > nul
echo =======================================================
echo 正在激活 Deeplearning 环境...
call conda activate Deeplearning
if %errorlevel% neq 0 (
    echo [错误] 无法激活 Deeplearning 环境！请确保您安装了 conda 并且该环境存在。
    pause
    exit /b 1
)

echo =======================================================
echo 开始运行量化交易全流程 (股票池-预处理-训练-回测)
echo =======================================================

echo [1/7] 正在构建股票池...
python scripts/build_stock_pool.py --date 20260527 --output outputs/stock_pool_20260527.csv --summary-output outputs/stock_pool_20260527.json
if %errorlevel% neq 0 goto :error

echo [2/7] 正在处理截面特征 (用于MLP)...
python scripts/preprocess_features.py
if %errorlevel% neq 0 goto :error

echo [3/7] 正在处理序列特征 (用于TSN)...
python scripts/preprocess_sequences.py
if %errorlevel% neq 0 goto :error

echo [4/7] 正在训练TSN+GRU主模型...
python scripts/train_tsn.py --data-dir outputs/preprocessed_seq --model-type tsn_gru
if %errorlevel% neq 0 goto :error

echo [5/7] 正在训练MLP基线模型...
python scripts/train_mlp_baseline.py --data-dir outputs/preprocessed
if %errorlevel% neq 0 goto :error

echo [6/7] 正在进行TSN+GRU模型历史回测(含手续费)...
python scripts/run_backtest.py --predictions outputs/predictions/tsn_valid_predictions.csv --output-dir outputs/backtest/tsn_valid
if %errorlevel% neq 0 goto :error

echo [7/7] 正在进行MLP模型历史回测(含手续费)...
python scripts/run_backtest.py --predictions outputs/predictions/mlp_valid_predictions.csv --output-dir outputs/backtest/mlp_valid
if %errorlevel% neq 0 goto :error

echo =======================================================
echo 所有流程已全部成功跑完！
echo 可以在 outputs/backtest/ 目录下查看回测效果。
echo =======================================================
pause
exit /b 0

:error
echo.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo 运行过程中出现错误，流程已中止！请检查上方报错信息。
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pause
exit /b 1
