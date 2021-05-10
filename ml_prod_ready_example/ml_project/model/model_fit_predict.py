import pickle
from typing import Dict
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ml_project.entities import ModelParams


def build_model(params: ModelParams) -> RandomForestClassifier:
    model = RandomForestClassifier(
        random_state=params.random_state,
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
    )
    return model


def build_pipeline(model: RandomForestClassifier, transformer: ColumnTransformer) -> Pipeline:
    return Pipeline([("feature_generation", transformer), ("clf", model)])


def train_model(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> Pipeline:
    model.fit(features, target)
    return model


def predict_model(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "r2_score": r2_score(target, predicts),
        "rmse": mean_squared_error(target, predicts, squared=False),
        "mae": mean_absolute_error(target, predicts),
    }


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
