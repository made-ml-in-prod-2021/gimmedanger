import click
import logging

import pandas as pd

from src import predict_model, deserialize_model

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


def predict_impl(path_to_model: str, path_to_input: str, path_to_output: str):

    logger.info(f'reading features at {path_to_input}')
    features = pd.read_csv(path_to_input)
    logger.info(f'read features shape = {features.shape}')

    logger.info(f'loading model at {path_to_model}')
    model = deserialize_model(path_to_model)
    logger.info(model)

    logger.info(f'start prediction')
    preds = predict_model(model, features)

    logger.info(f'saving predictions at {path_to_output}')
    preds.to_csv(path_to_output)


@click.command(name='predict')
@click.argument('path_to_model')
@click.argument('path_to_input')
@click.argument('path_to_output')
def predict_command(path_to_model: str, path_to_input: str, path_to_output: str):
    predict_impl(path_to_model, path_to_input, path_to_output)


if __name__ == '__main__':
    predict_command()
