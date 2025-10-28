import pandas as pd

from src.utils import read_results

df = read_results(read_len=True)
df = df[['MetaARIMA', 'AutoETS', 'series_length']]

ts_lens = [50, 75, 100, 125, 150, 175, 200, 300, 10000]
perf_by_max_length = []
for ts_len in ts_lens:
    df_ = df.query(f'series_length < {ts_len}')

    avg_smape = df_[['MetaARIMA', 'AutoETS']].mean()
    avg_smape['max_length'] = ts_len
    avg_smape['support'] = df_.shape[0] / df.shape[0]
    perf_by_max_length.append(avg_smape)

perf = pd.DataFrame(perf_by_max_length)
perf = perf.round(4)
perf['max_length'] = perf['max_length'].astype(int)
perf['support'] = perf['support'].round(2)
perf = perf.astype(str)

perf_tab = perf.to_latex(caption='CAPTION', label='tab:scores_by_maxlen')
print(perf_tab)
