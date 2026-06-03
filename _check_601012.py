"""Deep dive on 601012 隆基绿能 — why rank 1?"""
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 1. Price trend
print("=" * 60)
print("601012.SH (隆基绿能) 近期走势")
print("=" * 60)
dates = [20260508,20260511,20260512,20260513,20260514,20260515,
         20260518,20260519,20260520,20260521,20260522,20260525,
         20260526,20260527,20260528,20260529,20260601]

for date in dates:
    try:
        df = pd.read_csv(f'A股数据/daily/{date}.csv', dtype={'ts_code': str})
        r = df[df['ts_code'] == '601012.SH']
        if not r.empty:
            close = float(r['close'].values[0])
            vol = float(r['vol'].values[0]) if 'vol' in r.columns else 0
            amount = float(r['amount'].values[0]) if 'amount' in r.columns else 0
            pct = float(r['pct_chg'].values[0]) if 'pct_chg' in r.columns else 0
            bar = '█' * max(0, int(close * 2 - 20))
            print(f"  {date}  close={close:.2f}  chg={pct:+.2f}%  vol={vol/10000:.1f}万手  amt={amount/1e8:.2f}亿  {bar}")
    except Exception as e:
        print(f"  {date}  N/A - {e}")

# 2. Compare: model prediction vs actual return
print("\n" + "=" * 60)
print("模型预测 vs 实际走势")
print("=" * 60)
pred_files = ['outputs/predictions/latest_predictions_20260529.csv',
              'outputs/predictions/latest_predictions_20260601.csv']
for pf in pred_files:
    try:
        pred = pd.read_csv(pf, dtype={'ts_code': str})
        r = pred[pred['ts_code'] == '601012.SH']
        if not r.empty:
            rank_all = list(pred.sort_values('score', ascending=False)['ts_code']).index('601012.SH') + 1
            print(f"  {pf.split('_')[-1].replace('.csv','')}: score={r['score'].values[0]:.6f}  rank={rank_all}")
    except:
        pass

# 3. Get all available features for 601012 to understand what model sees
print("\n" + "=" * 60)
print("模型输入特征快照 (latest window)")
print("=" * 60)
try:
    from preprocess_seq import build_sequence_features_for_date
    from preprocess import available_daily_dates, finalized_feature_columns
    from stock_pool import build_stock_pool, DEFAULT_INDEX_CODE, DEFAULT_MIN_AMOUNT
    from pathlib import Path

    data_dir = Path('A股数据')
    trading_dates = available_daily_dates(data_dir)
    date_to_index = {d: i for i, d in enumerate(trading_dates)}

    # Latest available
    latest = trading_dates[-1]
    feature_end = trading_dates[-3]
    window_dates = trading_dates[-22:-2]

    pool, _ = build_stock_pool(data_dir, as_of_date=feature_end,
                                index_code=DEFAULT_INDEX_CODE, min_amount=DEFAULT_MIN_AMOUNT)

    seq = build_sequence_features_for_date(data_dir, pool, window_dates)
    feats = finalized_feature_columns(seq)

    r = seq[seq['ts_code'] == '601012.SH']
    if not r.empty:
        r_sorted = r.sort_values('trade_date')
        # Show last 5 days of key features
        key_feats = [c for c in feats if c in r.columns][:15]
        print(f"  特征维度: {len(feats)}")
        print(f"  window大小: {len(r_sorted)}")
        print(f"\n  最后5个交易日的部分特征:")
        print(f"  {'date':<10s} " + " ".join(f"{f:<10s}" for f in key_feats[:8]))
        for _, row in r_sorted.tail(5).iterrows():
            vals = " ".join(f"{float(row[f]):>10.4f}" if f in row and pd.notna(row[f]) else f"{'N/A':>10s}" for f in key_feats[:8])
            print(f"  {int(row['trade_date']):<10d} {vals}")
except Exception as e:
    print(f"  Feature analysis failed: {e}")

# 4. Sector context — are other solar stocks also highly ranked?
print("\n" + "=" * 60)
print("光伏/新能源同行业对比")
print("=" * 60)
try:
    basic = pd.read_csv('A股数据/basic.csv', usecols=['ts_code','name','industry'], dtype={'ts_code':str})
    solar_codes = basic[basic['industry'].str.contains('电气|光伏|新能源', na=False)]['ts_code'].tolist()

    pred = pd.read_csv('outputs/predictions/latest_predictions_20260601.csv', dtype={'ts_code': str})
    pred = pred.sort_values('score', ascending=False)

    for i, (_, row) in enumerate(pred.iterrows()):
        if row['ts_code'] in solar_codes:
            name = basic[basic['ts_code']==row['ts_code']]['name'].values
            name = name[0] if len(name) > 0 else '?'
            print(f"  rank={i+1:>3d}  {row['ts_code']} {name:<8s}  score={row['score']:.6f}")
except Exception as e:
    print(f"  Sector analysis failed: {e}")

# 5. Fundamental context — check market cap, PE etc if available
print("\n" + "=" * 60)
print("基本面参考")
print("=" * 60)
try:
    basic = pd.read_csv('A股数据/basic.csv', dtype={'ts_code':str})
    r = basic[basic['ts_code']=='601012.SH']
    if not r.empty:
        cols = [c for c in basic.columns if c in ['ts_code','name','industry','area','market','list_date']]
        for c in cols:
            print(f"  {c}: {r[c].values[0]}")
except Exception as e:
    print(f"  {e}")
