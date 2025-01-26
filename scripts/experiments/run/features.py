import pandas as pd
from cardtale.analytics.testing.card.trend import DifferencingTests

from utils.load_data.config import DATASETS, DATA_GROUPS, GROUP_IDX

# ---- data loading and partitioning
EXPERIMENT = 'features'
data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)

train, _ = data_loader.train_test_split(df, horizon=horizon)

# ---- feature extraction
features_l = []
for uid, uid_df in train.groupby('unique_id'):
    try:
        trend = DifferencingTests.ndiffs(uid_df['y'], test='kpss', test_type='level')
    except OverflowError:
        trend = 0

    seas = DifferencingTests.nsdiffs(uid_df['y'], test='seas', period=freq_int)

    trend_str = 'Non-stationary' if trend > 0 else 'Stationary'
    seas_str = 'Seasonal' if seas > 0 else 'Non-seasonal'

    features_l.append({
        'unique_id': uid,
        'trend_str': trend_str,
        'seas_str': seas_str,
    })

strength_df = pd.DataFrame(features_l)

strength_df.to_csv(f'assets/results/{data_name},{group},{EXPERIMENT}.csv', index=False)
