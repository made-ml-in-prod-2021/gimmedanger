import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Tuple

from ml_project.entities import read_training_pipeline_params, TrainingPipelineParams
from ml_project.features import build_transformer, split_features_target_data
from ml_project.grid_search import build_search_pipeline, run_grid_search
from ml_project.data import read_data, prepare_data, split_train_val_data
from ml_project.model import serialize_model, build_pipeline, build_model, train_model, predict_model, evaluate_model

import sys
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def run_train_grid_search_pipeline(data: pd.DataFrame, params: TrainingPipelineParams) \
        -> Tuple[RandomForestClassifier, Dict[str, float]]:

    transformer = build_transformer(params.feature_params)
    search_pipe = build_search_pipeline(transformer, params.model_params, params.search_params)
    logger.info(f'Grid search pipeline created!')

    features, target = split_features_target_data(data, params.feature_params)
    search = run_grid_search(search_pipe, features, target)
    logger.info(f'Grid search CV best score: {search.best_score_}')
    logger.info(f'Grid Search CV best params: {search.best_params_}')

    return search.best_estimator_, {'best_score': search.best_score_}


def run_train_pipeline(data: pd.DataFrame, params: TrainingPipelineParams) \
        -> Tuple[RandomForestClassifier, Dict[str, float]]:

    df_train, df_val = split_train_val_data(data, params.split_train_val_params)
    logger.info(f'Train dataset {df_train.shape}, Val dataset {df_val.shape} splitted!')

    transformer = build_transformer(params.feature_params)
    model = build_model(params.model_params)
    pipe = build_pipeline(model, transformer)
    logger.info(f'Training pipeline created!')

    x_train, y_train = split_features_target_data(df_train, params.feature_params)
    train_model(pipe, x_train, y_train)
    logger.info(f'Training pipeline finished!')

    x_val, y_val = split_features_target_data(df_val, params.feature_params)
    preds = predict_model(pipe, x_val)
    metrics = evaluate_model(preds, y_val)
    logger.info(f'Val metrics: {metrics}')

    return model, metrics


@hydra.main(config_path='../configs')
def run_pipeline(cfg: DictConfig) -> None:
    params = read_training_pipeline_params(cfg)
    logger.info(f'Loaded config: {params}')

    cwd_path = hydra.utils.get_original_cwd()
    load_path = f'{cwd_path}/{params.input_data_path}'
    df = prepare_data(read_data(load_path))[0]
    logger.info(f'Full dataset {df.shape} loaded: {load_path}')

    if params.run_grid_search:
        model, metrics = run_train_grid_search_pipeline(df, params)
    else:
        model, metrics = run_train_pipeline(df, params)

    save_path = f'{cwd_path}/{params.output_model_path}'
    serialize_model(model, save_path)
    logger.info(f'Trained model saved as pickle: {save_path}')


if __name__ == '__main__':
    run_pipeline()
