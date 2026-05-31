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
echo Quantitative Trading Pipeline
echo   Preprocess once -> Train eval model -> Train competition model -> Backtest
echo =======================================================

echo [1/5] Building stock pool...
python scripts/build_stock_pool.py --date 20260527 --output outputs/stock_pool_20260527.csv --summary-output outputs/stock_pool_20260527.json
if %errorlevel% neq 0 goto :error

echo [2/5] Preprocessing sequence features (one-time)...
echo      train=2019-2024  valid=2025  test=2026.1-5
python scripts/preprocess_sequences.py
if %errorlevel% neq 0 goto :error

echo [3/5] Training evaluation model (for report)...
echo      train on 2019-2024, validate on 2025
python scripts/train_tsn.py --data-dir outputs/preprocessed_seq --model-type tsn_gru
if %errorlevel% neq 0 goto :error

echo [4/5] Training competition model (for Jun 1-12 trading)...
echo      train on 2019-2026.5 (all data), hold out last 10pct for early-stop
python scripts/train_tsn.py --data-dir outputs/preprocessed_seq --model-type tsn_gru --include-valid
if %errorlevel% neq 0 goto :error

echo [5/5] Running backtest on evaluation model (valid period 2025)...
python scripts/run_backtest.py --predictions outputs/predictions/tsn_eval_valid_predictions.csv --output-dir outputs/backtest/tsn_eval_valid
if %errorlevel% neq 0 goto :error

echo =======================================================
echo All done!
echo   Evaluation backtest: outputs/backtest/tsn_eval_valid/
echo   Eval model:         outputs/models/tsn_eval_best.pt
echo   Competition model:  outputs/models/tsn_full_best.pt
echo =======================================================
echo.
echo For daily trading (Jun 1-12), run:
echo   python scripts/predict_latest.py --checkpoint outputs/models/tsn_full_best.pt
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
