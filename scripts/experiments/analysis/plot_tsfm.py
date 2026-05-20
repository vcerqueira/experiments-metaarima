import pandas as pd
import plotnine as p9

from src.chronos_data import ChronosDataset
from src.utils import THEME

algorithm = 'catboost'

target = 'monash_m3_monthly'

df, horizon, _, freq, seas_len = ChronosDataset.load_everything(target)
train, test = ChronosDataset.time_wise_split(df, horizon)

fcst = pd.read_csv('assets/results/tsfm/chronos2/predictions,monash_m3_monthly.csv')

uid = 'T001407'  # chronos best
# uid = 'T000816' # metaarima best
# uid = 'T001384' # metaarima greatest diff
# uid = 'T000696'  # chronos greatest diff

df_uid_tr = train.query(f'unique_id=="{uid}"').reset_index(drop=True).tail(24)
df_uid_ts = test.query(f'unique_id=="{uid}"').reset_index(drop=True)
fcst_uid = fcst.query(f'unique_id=="{uid}"').reset_index(drop=True)

df_uid_tr['ds'] = pd.to_datetime(df_uid_tr['ds'])
df_uid_ts['ds'] = pd.to_datetime(df_uid_ts['ds'])
fcst_uid['ds'] = pd.to_datetime(fcst_uid['ds'])

hist = df_uid_tr[['ds', 'y']].assign(series='Historical')
actual = df_uid_ts[['ds', 'y']].assign(series='Actual')
chronos = fcst_uid[['ds']].assign(y=fcst_uid['Chronos2'], series='Chronos2')
metaarima = fcst_uid[['ds']].assign(y=fcst_uid['MetaARIMA'], series='MetaARIMA')

plot_df = pd.concat([hist, actual, chronos, metaarima], ignore_index=True)
plot_df['series'] = pd.Categorical(
    plot_df['series'],
    categories=['Historical', 'Actual', 'MetaARIMA', 'Chronos2'],
)

plot = (
        p9.ggplot(plot_df, p9.aes(x='ds', y='y', color='series', linetype='series'))
        + THEME
        + p9.theme(
    plot_margin=0.015,
    axis_text_x=p9.element_text(size=18),
    axis_text_y=p9.element_text(size=18),
    axis_title_y=p9.element_text(size=16),
    legend_text=p9.element_text(size=18),
    legend_title=p9.element_blank(),
    legend_position='bottom',
)
        + p9.geom_line(size=1)
        + p9.scale_color_manual(
    values={
        'Historical': 'grey',
        'Actual': 'black',
        'MetaARIMA': 'steelblue',
        'Chronos2': 'orangered',
    }
)
        + p9.scale_linetype_manual(
    values={
        'Historical': 'solid',
        'Actual': 'solid',
        'MetaARIMA': 'dashed',
        'Chronos2': 'dashed',
    }
)
        + p9.labs(x='', y='value')
)

plot.save(f'assets/results/plots/tsfm_predictions_{uid}.pdf', width=12, height=4.5)
