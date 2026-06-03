import pandas as pd
basic = pd.read_csv('A股数据/basic.csv', usecols=['ts_code','name','industry'], dtype={'ts_code':str})
codes = ['600309.SH','600989.SH','688047.SH','000651.SZ','601012.SH','601698.SH','601872.SH','300418.SZ','002028.SZ','000858.SZ']
for code in codes:
    r = basic[basic['ts_code']==code]
    if not r.empty:
        row = r.iloc[0]
        print(f"{code}  {row['name']:<10s}  {row['industry']}")
