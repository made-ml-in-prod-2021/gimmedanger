import click
import logging
import yaml
import os

from omegaconf import DictConfig

from src import read_data, split_train_val_data, read_split_train_val_params

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


def split_impl(path_to_config: str, path_to_input: str, path_to_output: str):

    logger.info(f'loading config from {path_to_config}')
    with open(path_to_config, 'r') as f:
        raw_cfg = DictConfig(yaml.load(f, yaml.FullLoader))
        params = read_split_train_val_params(raw_cfg)
    logger.info(f'loaded config: {params}')

    logger.info(f'reading dataset at {path_to_input}')
    df = read_data(path_to_input)
    logger.info(f'read dataset shape = {df.shape}')

    logger.info(f'splitting dataset')
    train_df, val_df = split_train_val_data(df, params)
    logger.info(f'train dataset shape = {train_df.shape}')
    logger.info(f'val dataset shape = {val_df.shape}')

    logger.info(f'saving datasets at {path_to_output}')
    if not os.path.exists(path_to_output):
        os.mkdir(path_to_output)
    train_df.to_csv(f'{path_to_output}/train_data.csv')
    val_df.to_csv(f'{path_to_output}/val_data.csv')


@click.command(name='split')
@click.argument('path_to_config')
@click.argument('path_to_input')
@click.argument('path_to_output')
def split_command(path_to_config: str, path_to_input: str, path_to_output: str):
    split_impl(path_to_config, path_to_input, path_to_output)


if __name__ == '__main__':
    split_command()
