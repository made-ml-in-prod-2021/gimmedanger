import click
import logging
import os
import shutil

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


def prepare_data_impl(path_to_raw: str, path_to_output: str):

    logger.info(f'reading raw dataset at {path_to_raw}')
    if not os.path.exists(path_to_raw):
        raise RuntimeError('path to raw dataset is missing')

    logger.info(f'saving processed dataset at {path_to_output}')
    outdir, _, filename = path_to_output.rpartition('/')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    shutil.copyfile(path_to_raw, path_to_output)


@click.command(name='prepare_data')
@click.argument('path_to_raw')
@click.argument('path_to_output')
def prepare_data_command(path_to_raw: str, path_to_output: str):
    prepare_data_impl(path_to_raw, path_to_output)


if __name__ == '__main__':
    prepare_data_command()
