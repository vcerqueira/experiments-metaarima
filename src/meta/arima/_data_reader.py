from typing import Optional
import os
import joblib
import gzip
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd

from src.meta.arima._base import MetaARIMAUtils


class MetadataReader:
    def __init__(self, dataset_name: str,
                 group: str,
                 freq_int: int,
                 id_col: str = 'unique_id'):
        """
        Args:
            dataset_name: str, name of the dataset
            group: str, group of the dataset
            freq_int: int, frequency of the dataset
            id_col: str, name of the id column
        """
        load_dotenv()
        self.features_dir = Path(os.environ["FEATURES_DIR"])
        self.metadata_cv_dir = Path(os.environ["METADATA_CV_DIR"])
        self.id_col = id_col
        self.input_variables = None
        self.model_names = None
        self.dataset_name = dataset_name
        self.group = group
        self.freq_int = freq_int

    def read(self, from_dev_set: bool = False, fill_na_value: Optional[float] = -1):
        """ Read the data from the features and metadata cv files.
        Args:
            fill_na_value: float, value to fill na with
            fill_na_value is not None, then fill na with the value

        Returns:
            X: pd.DataFrame, features
            y: pd.DataFrame, target
            input_variables: list, input variables
            model_names: list, model names
            cv: pd.DataFrame, metadata cv
        """
        if from_dev_set:
            feats_path = self.features_dir / f"features-dev,{self.dataset_name},{self.group}.csv"
            cv_path = self.metadata_cv_dir / f"arima-dev,{self.dataset_name},{self.group}.csv"
        else:
            feats_path = self.features_dir / f"features,{self.dataset_name},{self.group}.csv"
            cv_path = self.metadata_cv_dir / f"arima,{self.dataset_name},{self.group}.csv"

        feats = pd.read_csv(feats_path)
        cv = pd.read_csv(cv_path)

        cv = cv.merge(feats, on=[self.id_col]).set_index(self.id_col)
        self.input_variables = feats.set_index(self.id_col).columns.tolist()
        self.model_names = MetaARIMAUtils.get_models_sf(season_length=self.freq_int, return_names=True)

        X = cv.loc[:, self.input_variables]
        if fill_na_value is not None:
            X = X.fillna(fill_na_value)
        y = cv.loc[:, self.model_names]

        return X, y, self.input_variables, self.model_names, cv


class ModelIO:
    @staticmethod
    def save_model(model, filename):
        with gzip.open(filename, 'wb') as f:
            joblib.dump(model, f)

    @staticmethod
    def load_model(filename):
        with gzip.open(filename, 'rb') as f:
            return joblib.load(f)
