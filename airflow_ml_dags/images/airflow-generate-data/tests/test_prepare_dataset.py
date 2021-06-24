import unittest

from typing import NoReturn
from omegaconf import DictConfig

from src \
    import read_generate_data_params, \
    prepare_data, read_data, generate_data, \
    ORIG_SHAPE, ORIG_COLUMNS, THALASSEMIA_TYPES, \
    SEX_TYPES, CHEST_PAIN_TYPES, FASTING_BLOOD_SUGAR_TYPES, \
    REST_ECG_TYPES, EXERCISE_INDUCED_ANGINA_TYPES, ST_SLOPE_TYPES, \
    FINAL_COLUMNS

PATH = f'tests/test_data.csv'


class PrepareDatasetTestCase(unittest.TestCase):
    def test_read_data(self) -> NoReturn:
        df = read_data(PATH)
        self.assertEqual(df.shape, ORIG_SHAPE)
        self.assertTupleEqual(tuple(df.columns), ORIG_COLUMNS)

    def test_prepare_data(self) -> NoReturn:
        df = read_data(PATH)
        mod_df = prepare_data(df)
        self.assertEqual(mod_df.shape, ORIG_SHAPE)
        self.assertTupleEqual(tuple(mod_df.columns), FINAL_COLUMNS)
        self.assertSetEqual(set(mod_df['thalassemia'].unique()), set(THALASSEMIA_TYPES))
        self.assertSetEqual(set(mod_df['sex'].unique()), set(SEX_TYPES))
        self.assertSetEqual(set(mod_df['chest_pain_type'].unique()), set(CHEST_PAIN_TYPES))
        self.assertSetEqual(set(mod_df['fasting_blood_sugar'].unique()), set(FASTING_BLOOD_SUGAR_TYPES))
        self.assertSetEqual(set(mod_df['rest_ecg'].unique()), set(REST_ECG_TYPES))
        self.assertSetEqual(set(mod_df['exercise_induced_angina'].unique()), set(EXERCISE_INDUCED_ANGINA_TYPES))
        self.assertSetEqual(set(mod_df['st_slope'].unique()), set(ST_SLOPE_TYPES))

    def test_generate_data(self) -> NoReturn:
        raw_cfg = DictConfig({'dataset_rows': 13,
                              'target_required': False,
                              'path_historical_data': PATH})
        params = read_generate_data_params(raw_cfg)
        feature_params = params.feature_params
        gen_df = generate_data(params)
        self.assertEqual(gen_df.shape[0], params.dataset_rows)
        expected_columns = set(feature_params.categorical_features).union(set(feature_params.numerical_features))
        self.assertSetEqual(set(gen_df.columns), expected_columns)


if __name__ == '__main__':
    unittest.main()
