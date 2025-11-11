# meta-arima

# experiments workflows

first, run scripts/metadata_collection

- scripts/metadata_collection/arima.py evaluates a large set (400) of configurations
- scripts/metadata_collection/feature_extraction.py extracts a feature set (tsfeeatures) from the corresponding time series

second, run scripts/experiments/metalearning to build and test the metaearning model

## todo

- run quantile and other sens analysis
- re.-implementar ablation
  - opt catboost class
- efficiency refactor
- redo analysis
- write paper
- +gifteval
- prod @ metaforecast

# contributions

- metaarima
  - multi-label dataset where target contains the top percentile of configurations
    - modeled using a classifier chain
    - predictors are based on features, tsfeatures
  - mmr re-ranking to improve diversity of selected configurations
- metadataset
- experiments and package


