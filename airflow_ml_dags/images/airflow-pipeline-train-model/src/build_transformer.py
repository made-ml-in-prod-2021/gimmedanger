import pandas as pd
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .feature_params import FeatureParams


def build_categorical_pipeline() -> Pipeline:
    return Pipeline(
        [
            ('OneHot', OneHotEncoder()),
        ]
    )


def build_numerical_pipeline() -> Pipeline:
    return Pipeline(
        [
            ('Normalize', MinMaxScaler()),
        ]
    )


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
        ],
        remainder='drop',
        sparse_threshold=1
    )


def split_features_target_data(df: pd.DataFrame, params: FeatureParams) -> Tuple[pd.DataFrame, pd.Series]:
    return df.drop(params.target_col, 1), df[params.target_col]
