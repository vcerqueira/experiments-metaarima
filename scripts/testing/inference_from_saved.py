import pandas as pd
import numpy as np

from src.meta.arima._data_reader import ModelIO

FILENAME = 'assets/trained_metaarima_m4m.joblib.gz'

metaarima = ModelIO.load_model(FILENAME)

n, freq, seas_l = 100, 'QE', 4
date_range = pd.date_range(start='2000-01-01', periods=n, freq=freq)
y_values = np.random.randn(n)

df = pd.DataFrame({
    'unique_id': ['X'] * n,
    'ds': date_range,
    'y': y_values
})

metaarima.fit(df, freq=freq, seas_length=seas_l)

print(metaarima.predict(h=18, level=None))
print(metaarima.predict(h=12, level=[95]))
