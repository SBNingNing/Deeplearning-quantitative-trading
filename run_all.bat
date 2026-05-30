@echo off
chcp 65001 > nul
echo =======================================================
echo Activating Deeplearning conda environment...
call conda activate Deeplearning
if %errorlevel% neq 0 (
    echo [ERROR] Cannot activate Deeplearning environment!
    pause
    exit /b 1
)

echo =======================================================
echo Quantitative Trading Pipeline (Pool-Preprocess-Train-Backtest)
echo =======================================================

echo [1/4] Building stock pool...
python scripts/build_stock_pool.py --date 20260527 --output outputs/stock_pool_20260527.csv --summary-output outputs/stock_pool_20260527.json
if %errorlevel% neq 0 goto :error

echo [2/4] Preprocessing sequence features (TSN+GRU)...
python scripts/preprocess_sequences.py
if %errorlevel% neq 0 goto :error

echo [3/4] Training TSN+GRU model...
python scripts/train_tsn.py --data-dir outputs/preprocessed_seq --model-type tsn_gru
if %errorlevel% neq 0 goto :error

echo [4/4] Running backtest (with transaction costs)...
python scripts/run_backtest.py --predictions outputs/predictions/tsn_valid_predictions.csv --output-dir outputs/backtest/tsn_valid
if %errorlevel% neq 0 goto :error

echo =======================================================
echo All done! Check outputs/backtest/tsn_valid for results.
echo =======================================================
pause
exit /b 0

:error
echo.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo ERROR: Pipeline aborted. Check the messages above.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pause
exit /b 1
