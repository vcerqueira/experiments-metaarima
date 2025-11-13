# meta-arima

# experiments workflows

first, run scripts/metadata_collection

- scripts/metadata_collection/arima.py evaluates a large set (400) of configurations
- scripts/metadata_collection/feature_extraction.py extracts a feature set (tsfeeatures) from the corresponding time series

second, run scripts/experiments/metalearning to build and test the metaearning model

## todo

- write paper
- gifteval
- prod @ metaforecast


