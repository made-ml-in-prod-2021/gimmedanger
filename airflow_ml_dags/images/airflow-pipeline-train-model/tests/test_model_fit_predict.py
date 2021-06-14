import unittest
import pandas as pd

from sklearn.utils.validation import check_is_fitted

from src import build_transformer, split_features_target_data
from src import FeatureParams, ModelParams
from src import build_model, build_pipeline, train_model, evaluate_model, predict_model

PATH = f'tests/test_data_prepared.csv'


class TestModelFitPredictTestCase(unittest.TestCase):
    def test(self):
        feature_params = FeatureParams(
            numerical_features=[
                'cholesterol', 'age', 'st_depression',
                'max_heart_rate_achieved', 'resting_blood_pressure',
                'num_major_vessels'
            ],
            categorical_features=[
                'sex', 'chest_pain_type', 'fasting_blood_sugar',
                'rest_ecg', 'exercise_induced_angina', 'st_slope',
                'thalassemia'
            ],
            target_col='target'
        )
        model_params = ModelParams(
            random_state=1024,
            n_estimators=100,
            max_depth=5
        )
        df = pd.read_csv(PATH)
        features, target = split_features_target_data(df, feature_params)
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
