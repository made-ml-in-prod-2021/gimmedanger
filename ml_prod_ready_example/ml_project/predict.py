import hydra
import pickle
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Tuple

from ml_project.entities import read_predict_command_params, PredictCommandParams
from ml_project.features import build_transformer, split_features_target_data
from ml_project.grid_search import build_search_pipeline, run_grid_search
from ml_project.data import read_data, prepare_data, split_train_val_data
from ml_project.model import deserialize_model, build_pipeline, build_model, train_model, predict_model, evaluate_model

import sys
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

@hydra.main(config_path='../configs')
def predict_command(cfg: DictConfig) -> None:
    params = read_predict_command_params(cfg)
    logger.info(f'Loaded config: {params}')

    cwd_path = hydra.utils.get_original_cwd()
    data_path = f'{cwd_path}/{params.input_data_path}'
    df = prepare_data(read_data(data_path))[0]
    features, _ = split_features_target_data(df, params.feature_params)
    logger.info(f'Test dataset {features.shape} loaded: {data_path}')

    model_path = f'{cwd_path}/{params.trained_model_path}'
    model = deserialize_model(model_path)
    logger.info(f'Model {model} loaded: {model_path}')

    preds = predict_model(model, features)
    preds_path = f'{cwd_path}/{params.output_preds_path}'
    with open(preds_path, 'wb') as f:
        pickle.dump(preds, f)
    logger.info(f'Preds saved as pickle: {preds_path}')


if __name__ == '__main__':
    predict_command()