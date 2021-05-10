import unittest
import pandas as pd
from typing import NoReturn

from ml_project.entities import FeatureParams
from ml_project.data import FILENAME, read_data, prepare_data
from ml_project.features.build_transformer import build_transformer, build_numerical_pipeline, \
    build_categorical_pipeline


class BuildTransformerTestCase(unittest.TestCase):
    def test_categorical_features_pipeline(self) -> NoReturn:
        df = read_data(f'ml_project/data/{FILENAME}').head()
        df, all_cat_features, all_num_features = prepare_data(df)
        params = FeatureParams(categorical_features=['sex'])
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
        df = read_data(f'ml_project/data/{FILENAME}')
        df, all_cat_features, all_num_features = prepare_data(df)
        params = FeatureParams()
        df_num = df[params.numerical_features]
        pipe = build_numerical_pipeline()
        df_num_transformed = pd.DataFrame(pipe.fit_transform(df_num))
        for idx, name in enumerate(params.numerical_features):
            self.assertNotEqual(df_num.max()[idx], 1.)
            self.assertAlmostEqual(df_num_transformed.max()[idx], 1.)
            self.assertAlmostEqual(df_num_transformed.min()[idx], 0.)

    def test_full_features_pipeline(self) -> NoReturn:
        df = read_data(f'ml_project/data/{FILENAME}')
        df, all_cat_features, all_num_features = prepare_data(df)
        params = FeatureParams()
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
