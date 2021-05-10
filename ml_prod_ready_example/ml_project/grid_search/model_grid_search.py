import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ml_project.entities import ModelParams, SearchParams


def build_search_pipeline(transformer: ColumnTransformer, model_params: ModelParams,
                          search_params: SearchParams) -> GridSearchCV:
    parameters = {
        'clf__n_estimators': search_params.n_estimator_lst,
        'clf__max_depth': search_params.max_depth_lst,
        'clf__random_state': [model_params.random_state]
    }
    model = RandomForestClassifier()
    pipe = Pipeline([("feature_generation", transformer), ("clf", model)])
    return GridSearchCV(pipe, parameters)


def run_grid_search(search: GridSearchCV, features: pd.DataFrame, target: pd.Series) -> GridSearchCV:
    search.fit(features, target)
    return search
