import pandas as pd
import numpy as np

from src.meta.arima._data_reader import ModelIO

FILENAME = 'assets/trained_metaarima_m4m.joblib.gz'

metaarima = ModelIO.load_model(FILENAME)

date_range = pd.date_range(start='2000-01-01', periods=100, freq='M')
y_values = np.random.randn(100)

df = pd.DataFrame({
    'unique_id': ['X'] * 100,
    'ds': date_range,
    'y': y_values
})

freq = 12  # monthly

metaarima.fit_model(df, freq)

print(metaarima.predict(h=18, level=None))
print(metaarima.predict(h=12, level=[95]))
