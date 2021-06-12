import unittest
from typing import NoReturn
from omegaconf import DictConfig

from src import SplitTrainValParams, read_data, split_train_val_data, read_split_train_val_params

PATH = f'tests/test_data.csv'


class SplitDatasetTestCase(unittest.TestCase):
    def test(self) -> NoReturn:
        df = read_data(PATH)
        raw_cfg = DictConfig({'test_size': 0.1,
                              'random_state': 1024})
        params = read_split_train_val_params(raw_cfg)
        train_df, val_df = split_train_val_data(df, params)
        self.assertEqual(len(val_df) / len(train_df), 0.11397058823529412)


if __name__ == '__main__':
    unittest.main()
