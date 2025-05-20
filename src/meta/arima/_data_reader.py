import os
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd

from src.meta.arima._base import MetaARIMAUtils


class MetadataReader:
    @staticmethod
    def get_features_dir():
        load_dotenv()
        return Path(os.getenv("FEATURES_DIR", "assets/features/"))

    @staticmethod
    def get_metadata_cv_dir():
        load_dotenv()
        return Path(os.getenv("METADATA_CV_DIR", "assets/metadata_cv/"))

    @staticmethod
    def read(data_name, group, freq_int, features_dir=None, metadata_cv_dir=None):
        features_dir = Path(features_dir) if features_dir is not None else MetadataReader.get_features_dir()
        metadata_cv_dir = Path(metadata_cv_dir) if metadata_cv_dir is not None else MetadataReader.get_metadata_cv_dir()
        feats_path = features_dir / f"features,{data_name},{group}.csv"
        cv_path = metadata_cv_dir / f"arima,{data_name},{group}.csv"
        feats = pd.read_csv(feats_path)
        cv = pd.read_csv(cv_path)
        cv = cv.merge(feats, on=['unique_id']).set_index('unique_id')
        input_variables = feats.set_index('unique_id').columns.tolist()
        model_names = MetaARIMAUtils.get_models_sf(season_length=freq_int, return_names=True)
        X = cv.loc[:, input_variables].fillna(-1)
        y = cv.loc[:, model_names]
        return X, y, input_variables, model_names, cv
