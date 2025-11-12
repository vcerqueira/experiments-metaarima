from src.meta.arima._data_reader import ModelIO
from src.chronos_data import ChronosDataset

OVERRIDE_DS = False
algorithm = 'catboost'
source = 'm4_monthly'
FILENAME = f'assets/trained_metaarima_{source}_{algorithm}.joblib.gz'
meta_arima = ModelIO.load_model(FILENAME)

target = 'monash_m3_monthly'
# target = 'monash_hospital'
df, horizon, _, freq, seas_len = ChronosDataset.load_everything(target)
train, test = ChronosDataset.time_wise_split(df, horizon)

uids = train['unique_id'].unique().tolist()

tot_obs_metaarima, tot_obs_autoarima = 0, 0
for uid in uids:
    print(uid)

    df_uid_tr = train.query(f'unique_id=="{uid}"').reset_index(drop=True)
    df_uid_ts = test.query(f'unique_id=="{uid}"').reset_index(drop=True)

    meta_arima.fit(df_uid_tr, freq=freq, seas_length=seas_len)

    tot_obs_autoarima += df_uid_tr.shape[0] * 94
    tot_obs_metaarima += meta_arima.model.tot_nobs
    tot_obs_metaarima += df_uid_tr.shape[0]  # for the feature extraction step

print('metaarima tot:', tot_obs_metaarima)
print('autoarima tot:', tot_obs_autoarima)
