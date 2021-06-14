import unittest
import pandas as pd
from typing import NoReturn

from src import FeatureParams
from src import build_transformer, build_numerical_pipeline, build_categorical_pipeline


PATH = f'tests/test_data_prepared.csv'

class BuildTransformerTestCase(unittest.TestCase):
    def test_categorical_features_pipeline(self) -> NoReturn:
        df = pd.read_csv(PATH)
        params = FeatureParams(
            categorical_features=['sex'],
            numerical_features=[],
            target_col='target'
        )
        df_cat = df[params.categorical_features]
        pipe = build_categorical_pipeline()
        df_cat_transformed = pd.DataFrame(pipe.fit_transform(df_cat))
        for idx, exp_val in enumerate(['male', 'male', 'female', 'male', 'female']):
            self.assertEqual(df_cat.at[idx, 'sex'], exp_val)
            transformed_val = df_cat_transformed.at[idx, 0].toarray()[0]
            if exp_val == 'male':
                self.assertEqual(transformed_val[0], 0.)
                self.assertEqual(transformed_val[1], 1.)
            else:
                self.assertEqual(transformed_val[0], 1.)
                self.assertEqual(transformed_val[1], 0.)

    def test_numerical_features_pipeline(self) -> NoReturn:
        df = pd.read_csv(PATH)
        params = FeatureParams(numerical_features=[
            'cholesterol', 'age', 'st_depression',
            'max_heart_rate_achieved', 'resting_blood_pressure',
            'num_major_vessels'],
            categorical_features=[],
            target_col='target'
        )
        df_num = df[params.numerical_features]
        pipe = build_numerical_pipeline()
        df_num_transformed = pd.DataFrame(pipe.fit_transform(df_num))
        for idx, name in enumerate(params.numerical_features):
            self.assertNotEqual(df_num.max()[idx], 1.)
            self.assertAlmostEqual(df_num_transformed.max()[idx], 1.)
            self.assertAlmostEqual(df_num_transformed.min()[idx], 0.)

    def test_full_features_pipeline(self) -> NoReturn:
        df = pd.read_csv(PATH)
        params = FeatureParams(
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
        pipe = build_transformer(params)
        df_trans = pipe.fit_transform(df)
        new_columns_num = 0
        for cat_col in params.categorical_features:
            new_columns_num += len(df[cat_col].unique())
        new_columns_num += len(params.numerical_features)
        self.assertNotEqual(df.shape[1], new_columns_num)
        self.assertEqual(df_trans.shape[1], new_columns_num)


if __name__ == '__main__':
    unittest.main()
