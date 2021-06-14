import pickle
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

from .feature_params import FeatureParams


def split_features_target_data(df: pd.DataFrame, params: FeatureParams) -> Tuple[pd.DataFrame, pd.Series]:
    return df.drop(params.target_col, 1), df[params.target_col]


def predict_model(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        'r2_score': r2_score(target, predicts),
        'rmse': mean_squared_error(target, predicts, squared=False),
        'mae': mean_absolute_error(target, predicts),
    }


def deserialize_model(path: str) -> str:
    with open(path, 'rb') as f:
        return pickle.load(f)
