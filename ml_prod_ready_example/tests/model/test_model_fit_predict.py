import unittest

from sklearn.utils.validation import check_is_fitted

from ml_project.features import build_transformer, split_features_target_data
from ml_project.entities.feature_params import FeatureParams
from ml_project.entities.model_params import ModelParams
from ml_project.model import build_model, build_pipeline, train_model, evaluate_model, predict_model
from ml_project.data import read_data, prepare_data


class TestModelFitPredictTestCase(unittest.TestCase):
    def test(self):
        feature_params = FeatureParams()
        model_params = ModelParams()
        df = read_data(f'data/raw/dataset.csv')
        features, target = split_features_target_data(prepare_data(df)[0], feature_params)
        transformer = build_transformer(feature_params)
        model = build_model(model_params)
        pipe = build_pipeline(model, transformer)
        train_model(pipe, features, target)
        preds = predict_model(pipe, features)
        metrics = evaluate_model(preds, target)
        self.assertLess(0, metrics['r2_score'])
        self.assertLess(0, metrics['rmse'])
        self.assertLess(0, metrics['mae'])
        try:
            check_is_fitted(model)
        except:
            self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
