import unittest
from typing import NoReturn

from ml_project.entities.feature_params import FeatureParams
from ml_project.entities import TrainingPipelineParams
from ml_project.train_pipeline import run_train_grid_search_pipeline, run_train_pipeline
from ml_project.data import prepare_data, read_data


class TestFullGridSearchPipelineCase(unittest.TestCase):
    def test(self) -> NoReturn:
        params = TrainingPipelineParams()
        df = prepare_data(read_data(f'data/raw/dataset.csv'))[0]
        _, score = run_train_grid_search_pipeline(df, params)
        self.assertLess(0, score['best_score'])


class TestTrainPipelineCase(unittest.TestCase):
    def test(self) -> NoReturn:
        params = TrainingPipelineParams()
        df = prepare_data(read_data(f'data/raw/dataset.csv'))[0]
        _, score = run_train_pipeline(df, params)
        self.assertLess(0, score['r2_score'])


if __name__ == '__main__':
    unittest.main()
