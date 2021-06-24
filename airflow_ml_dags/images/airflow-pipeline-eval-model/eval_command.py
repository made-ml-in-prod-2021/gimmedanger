import click
import logging
import yaml

import pandas as pd
from omegaconf import DictConfig

from src import split_features_target_data, read_training_pipeline_params
from src import predict_model, evaluate_model, deserialize_model

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


def eval_impl(path_to_config: str, path_to_model: str, path_to_input: str, path_to_output: str):

    logger.info(f'loading config from {path_to_config}')
    with open(path_to_config, 'r') as f:
        raw_cfg = DictConfig(yaml.load(f, yaml.FullLoader))
        params = read_training_pipeline_params(raw_cfg)
    logger.info(f'loaded config: {params}')

    logger.info(f'reading val dataset at {path_to_input}')
    val_df = pd.read_csv(path_to_input)
    logger.info(f'read val dataset shape = {val_df.shape}')

    logger.info(f'loading model at {path_to_model}')
    model = deserialize_model(path_to_model)
    logger.info(model)

    logger.info(f'start evaluating')
    x_val, y_val = split_features_target_data(val_df, params.feature_params)
    preds = predict_model(model, x_val)
    metrics = evaluate_model(preds, y_val)
    logger.info(f'val metrics: {metrics}')

    logger.info(f'saving metrics at {path_to_output}')
    with open(path_to_output, 'w') as f:
        print(metrics, file=f)


@click.command(name='eval')
@click.argument('path_to_config')
@click.argument('path_to_model')
@click.argument('path_to_input')
@click.argument('path_to_output')
def eval_command(path_to_config: str, path_to_model: str, path_to_input: str, path_to_output: str):
    eval_impl(path_to_config, path_to_model, path_to_input, path_to_output)


if __name__ == '__main__':
    eval_command()
