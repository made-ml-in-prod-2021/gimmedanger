import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

from .split_train_val_params import SplitTrainValParams


def read_data(path: str) -> pd.DataFrame:
    """
    read_csv wrapper with asserts
    :param path: dataset path
    :return: loaded dataframe
    """
    return pd.read_csv(path)


def split_train_val_data(data: pd.DataFrame, params: SplitTrainValParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param data: full dataframe
    :param params: splitting parameters
    :return: train dataframe, val dataframe
    """
    if params.test_size == 0.0:
        return data, pd.DataFrame()
    train_data, val_data = train_test_split(
        data, test_size=params.test_size, random_state=params.random_state
    )
    return train_data, val_data
