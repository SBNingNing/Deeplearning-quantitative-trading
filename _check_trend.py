"""Check 600309 recent price trend and model prediction details."""
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 1. Price trend of 600309 over recent trading days
print("=" * 60)
print("600309.SH (万华化学) 近期走势")
print("=" * 60)
dates = [20260508,20260511,20260512,20260513,20260514,20260515,
         20260518,20260519,20260520,20260521,20260522,20260525,
         20260526,20260527,20260528,20260529]
prices = []
for date in dates:
    try:
        df = pd.read_csv(f'A股数据/daily/{date}.csv', dtype={'ts_code': str})
        r = df[df['ts_code'] == '600309.SH']
        if not r.empty:
            close = float(r['close'].values[0])
            pct = float(r['pct_chg'].values[0]) if 'pct_chg' in r.columns else 0
            prices.append((date, close, pct))
    except:
        pass

for date, close, pct in prices:
    bar = '█' * max(0, int((close - 70) * 2)) if close > 70 else ''
    print(f"  {date}  close={close:.2f}  chg={pct:+.2f}%  {bar}")

# 2. Compare with other holdings
print("\n" + "=" * 60)
print("持仓股票近期表现对比 (20260508 → 20260529)")
print("=" * 60)
holdings = ['600309.SH','600989.SH','688047.SH','000651.SZ','601012.SH',
            '601698.SH','601872.SH','300418.SZ','002028.SZ','000858.SZ']
basic = pd.read_csv('A股数据/basic.csv', usecols=['ts_code','name'], dtype={'ts_code':str})
name_map = dict(zip(basic['ts_code'], basic['name']))

for code in holdings:
    try:
        start_df = pd.read_csv(f'A股数据/daily/20260508.csv', dtype={'ts_code': str})
        end_df = pd.read_csv(f'A股数据/daily/20260529.csv', dtype={'ts_code': str})
        s = start_df[start_df['ts_code'] == code]
        e = end_df[end_df['ts_code'] == code]
        if not s.empty and not e.empty:
            s_close = float(s['close'].values[0])
            e_close = float(e['close'].values[0])
            ret = (e_close - s_close) / s_close * 100
            name = name_map.get(code, '?')
            print(f"  {code} {name:<8s}  {s_close:>8.2f} → {e_close:>8.2f}  {ret:+.2f}%")
    except:
        pass

# 3. Check if the model scores reflect this
print("\n" + "=" * 60)
print("最新预测排名 (signal date: 20260529)")
print("=" * 60)
pred_path = 'outputs/predictions/latest_predictions_20260529.csv'
try:
    pred = pd.read_csv(pred_path, dtype={'ts_code': str})
    pred = pred.sort_values('score', ascending=False)
    # Show top 15 and highlight holdings
    holdings_set = set(holdings)
    for i, (_, row) in enumerate(pred.head(30).iterrows()):
        code = row['ts_code']
        name = name_map.get(code, '?')
        marker = ' <-- 持仓' if code in holdings_set else ''
        print(f"  rank={i+1:>3d}  {code} {name:<8s}  score={row['score']:.6f}{marker}")
except Exception as e:
    print(f"  Prediction file not available: {e}")

# 4. Check volatility of 600309 vs others
print("\n" + "=" * 60)
print("近期波动率对比 (20日)")
print("=" * 60)
import numpy as np
for code in holdings:
    rets = []
    for i in range(1, len(dates)):
        try:
            d1 = pd.read_csv(f'A股数据/daily/{dates[i-1]}.csv', dtype={'ts_code': str})
            d2 = pd.read_csv(f'A股数据/daily/{dates[i]}.csv', dtype={'ts_code': str})
            p1 = d1[d1['ts_code'] == code]
            p2 = d2[d2['ts_code'] == code]
            if not p1.empty and not p2.empty:
                r = float(p2['close'].values[0]) / float(p1['close'].values[0]) - 1
                rets.append(r)
        except:
            pass
    if rets:
        vol = np.std(rets, ddof=0) * 100
        avg_ret = np.mean(rets) * 100
        name = name_map.get(code, '?')
        print(f"  {code} {name:<8s}  daily_vol={vol:.3f}%  avg_ret={avg_ret:+.3f}%")
