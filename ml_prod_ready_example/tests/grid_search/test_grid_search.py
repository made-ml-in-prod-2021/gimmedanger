import unittest

from sklearn.utils.validation import check_is_fitted

from ml_project.features import build_transformer, split_features_target_data
from ml_project.entities.feature_params import FeatureParams
from ml_project.entities.model_params import SearchParams, ModelParams
from ml_project.grid_search import build_search_pipeline, run_grid_search
from ml_project.data import FILENAME, read_data, prepare_data


class TestGridSearchTestCase(unittest.TestCase):
    def test(self):
        feature_params = FeatureParams()
        model_params = ModelParams()
        search_params = SearchParams(
            n_estimator_lst=[model_params.n_estimators],
            max_depth_lst=[model_params.max_depth]
        )
        df = read_data(f'ml_project/data/{FILENAME}')
        features, target = split_features_target_data(prepare_data(df)[0], feature_params)
        transformer = build_transformer(feature_params)
        search_pipe = build_search_pipeline(transformer, model_params, search_params)
        search = run_grid_search(search_pipe, features, target)
        self.assertLessEqual(0.8, search.best_score_)
        try:
            check_is_fitted(search)
        except:
            self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
