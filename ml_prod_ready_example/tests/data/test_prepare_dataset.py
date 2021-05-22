import unittest

from typing import NoReturn

from ml_project.entities import SplitTrainValParams

from ml_project.data \
    import prepare_data, read_data, split_train_val_data, \
    ORIG_SHAPE, ORIG_COLUMNS, THALASSEMIA_TYPES, \
    SEX_TYPES, CHEST_PAIN_TYPES, FASTING_BLOOD_SUGAR_TYPES, \
    REST_ECG_TYPES, EXERCISE_INDUCED_ANGINA_TYPES, ST_SLOPE_TYPES, \
    FINAL_COLUMNS, FINAL_CAT_COLUMNS, FINAL_NUM_COLUMNS


class PrepareDatasetTestCase(unittest.TestCase):
    def test_read_data(self) -> NoReturn:
        df = read_data(f'data/raw/dataset.csv')
        self.assertEqual(df.shape, ORIG_SHAPE)
        self.assertTupleEqual(tuple(df.columns), ORIG_COLUMNS)

    def test_prepare_data(self) -> NoReturn:
        df = read_data(f'data/raw/dataset.csv')
        mod_df, cat_features, num_features = prepare_data(df)
        self.assertEqual(mod_df.shape, ORIG_SHAPE)
        self.assertTupleEqual(tuple(mod_df.columns), FINAL_COLUMNS)
        self.assertTupleEqual(cat_features, FINAL_CAT_COLUMNS)
        self.assertTupleEqual(num_features, FINAL_NUM_COLUMNS)
        self.assertSetEqual(set(mod_df['thalassemia'].unique()), set(THALASSEMIA_TYPES))
        self.assertSetEqual(set(mod_df['sex'].unique()), set(SEX_TYPES))
        self.assertSetEqual(set(mod_df['chest_pain_type'].unique()), set(CHEST_PAIN_TYPES))
        self.assertSetEqual(set(mod_df['fasting_blood_sugar'].unique()), set(FASTING_BLOOD_SUGAR_TYPES))
        self.assertSetEqual(set(mod_df['rest_ecg'].unique()), set(REST_ECG_TYPES))
        self.assertSetEqual(set(mod_df['exercise_induced_angina'].unique()), set(EXERCISE_INDUCED_ANGINA_TYPES))
        self.assertSetEqual(set(mod_df['st_slope'].unique()), set(ST_SLOPE_TYPES))

    def test_split_train_val_data(self) -> NoReturn:
        df = read_data(f'data/raw/dataset.csv')
        # regular case
        params = SplitTrainValParams(test_size=0.1)
        train_df, val_df = split_train_val_data(df, params)
        self.assertEqual(len(val_df) / len(train_df), 0.11397058823529412)
        # grid search case
        params = SplitTrainValParams(test_size=0.0)
        train_df, val_df = split_train_val_data(df, params)
        self.assertEqual(len(val_df) / len(train_df), params.test_size)


if __name__ == '__main__':
    unittest.main()
