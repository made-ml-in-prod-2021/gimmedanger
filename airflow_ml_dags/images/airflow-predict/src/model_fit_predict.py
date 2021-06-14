import pickle
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline


def predict_model(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def deserialize_model(path: str) -> str:
    with open(path, 'rb') as f:
        return pickle.load(f)
