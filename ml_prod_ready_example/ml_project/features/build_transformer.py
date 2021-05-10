from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ml_project.entities import FeatureParams


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
        [
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
        ]
    )
