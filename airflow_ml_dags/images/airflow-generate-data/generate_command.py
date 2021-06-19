import click
import logging
import yaml
import os

from omegaconf import DictConfig

from src import generate_data
from src import read_generate_data_params

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


def generate_data_impl(path_to_config: str, path_to_output: str):

    logger.info(f'loading config from {path_to_config}')
    with open(path_to_config, 'r') as f:
        raw_cfg = DictConfig(yaml.load(f, yaml.FullLoader))
        params = read_generate_data_params(raw_cfg)
    logger.info(f'loaded config: {params}')

    logger.info(f'generating dataset')
    gen_df = generate_data(params)
    logger.info(f'generated dataset shape = {gen_df.shape}')

    logger.info(f'saving dataset at {path_to_output}')
    outdir, _, filename = path_to_output.rpartition('/')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    gen_df.to_csv(path_to_output)
    pred_df = gen_df.drop(columns=[params.feature_params.target_col])
    pred_df.to_csv(f'{outdir}/predict_data.csv')


@click.command(name='generate_data')
@click.argument('path_to_config')
@click.argument('path_to_output')
def generate_data_command(path_to_config: str, path_to_output: str):
    generate_data_impl(path_to_config, path_to_output)


if __name__ == '__main__':
    generate_data_command()
