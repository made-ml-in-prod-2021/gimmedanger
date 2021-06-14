import click
import logging
import yaml
import os

import pandas as pd
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple

from src import build_transformer, split_features_target_data
from src import read_training_pipeline_params, TrainingPipelineParams
from src import serialize_model, build_pipeline, build_model, train_model

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


def run_train_pipeline(
        df_train: pd.DataFrame,
        params: TrainingPipelineParams
) -> Tuple[Pipeline, Dict[str, float]]:

    transformer = build_transformer(params.feature_params)
    model = build_model(params.model_params)
    pipe = build_pipeline(model, transformer)
    logger.info(f'training pipeline created!')

    x_train, y_train = split_features_target_data(df_train, params.feature_params)
    train_model(pipe, x_train, y_train)
    logger.info(f'training pipeline finished!')

    return pipe


def train_impl(path_to_config: str, path_to_input: str, path_to_output: str):

    logger.info(f'loading config from {path_to_config}')
    with open(path_to_config, 'r') as f:
        raw_cfg = DictConfig(yaml.load(f, yaml.FullLoader))
        params = read_training_pipeline_params(raw_cfg)
    logger.info(f'loaded config: {params}')

    logger.info(f'reading train/val dataset at {path_to_input}')
    train_df = pd.read_csv(path_to_input)
    logger.info(f'read train dataset shape = {train_df.shape}')

    model = run_train_pipeline(train_df, params)
    logger.info(f'saving model at {path_to_output}')
    serialize_model(model, path_to_output)


@click.command(name='train')
@click.argument('path_to_config')
@click.argument('path_to_input')
@click.argument('path_to_output')
def train_command(path_to_config: str, path_to_input: str, path_to_output: str):
    train_impl(path_to_config, path_to_input, path_to_output)


if __name__ == '__main__':
    train_command()
