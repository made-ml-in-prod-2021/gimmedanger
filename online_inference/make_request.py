import requests
import click
import logging
import urllib
import json

from app import XInput, YResponse
from src.data import read_data

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


def make_predict_request(server_address: str,
                         path_to_data: str,
                         path_for_save_result: str):

    logger.info(f"read data")
    data = read_data(path_to_data)
    data_for_predict = XInput(data=data.values.tolist(), features=data.columns.tolist())

    logger.info(f"send request")
    response = requests.get(
        urllib.parse.urljoin(server_address, "predict"),
        json=data_for_predict.dict(),
    )

    logger.info(f"parse result")
    result = []
    if response.status_code == 200:
        for elem in response.json():
            predict = YResponse(**elem)
            result.append(predict.predict)
    else:
        result = response.json()

    logger.info(f"save result")
    with open(path_for_save_result, "w") as result_file:
        json.dump(result, result_file)


@click.command(name="make_predict")
@click.argument("server_address")
@click.argument("path_to_data")
@click.argument("path_for_save_result")
def make_predict_command(server_address: str,
                         path_to_data: str,
                         path_for_save_result: str):
    make_predict_request(server_address, path_to_data, path_for_save_result)


if __name__ == "__main__":
    make_predict_command()
