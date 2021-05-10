from typing import Tuple
from sklearn.model_selection import train_test_split
from ml_project.entities import SplitTrainValParams
import pandas as pd


FILENAME = 'dataset.csv'

TARGET_COLUMN = 'target'

ORIG_SHAPE = (303, 14)

ORIG_COLUMNS = ('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target')

SEX_TYPES = ('female',
             'male')

CHEST_PAIN_TYPES = ('unknown angina',
                    'typical angina',
                    'atypical angina',
                    'non-anginal pain')

FASTING_BLOOD_SUGAR_TYPES = ('lower than 120mg/ml',
                             'greater than 120mg/ml')

REST_ECG_TYPES = ('normal',
                  'ST-T wave abnormality',
                  'left ventricular hypertrophy')

EXERCISE_INDUCED_ANGINA_TYPES = ('yes',
                                 'no')

ST_SLOPE_TYPES = ('unknown',
                  'upsloping',
                  'flat')

THALASSEMIA_TYPES = ('unknown',
                     'normal',
                     'fixed defect',
                     'reversable defect')

FINAL_COLUMNS = ('age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
                 'cholesterol', 'fasting_blood_sugar', 'rest_ecg',
                 'max_heart_rate_achieved', 'exercise_induced_angina',
                 'st_depression', 'st_slope', 'num_major_vessels',
                 'thalassemia', 'target')

FINAL_CAT_COLUMNS = ('sex', 'chest_pain_type', 'fasting_blood_sugar',
                     'rest_ecg', 'exercise_induced_angina',
                     'st_slope', 'thalassemia')

FINAL_NUM_COLUMNS = tuple(set(FINAL_COLUMNS) - set(FINAL_CAT_COLUMNS) - {'target'})


def read_data(path: str) -> pd.DataFrame:
    """
    read_csv wrapper with asserts
    :param path: dataset path
    :return: loaded dataframe
    """
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, tuple, tuple]:
    """
    :param df: original dataframe
    :return: (modified dataframe, categorical features names, numerical features names)
    """
    df.columns = FINAL_COLUMNS

    for idx in range(len(SEX_TYPES)):
        df.loc[df['sex'] == idx, 'sex'] = SEX_TYPES[idx]

    for idx in range(len(CHEST_PAIN_TYPES)):
        df.loc[df['chest_pain_type'] == idx, 'chest_pain_type'] = CHEST_PAIN_TYPES[idx]

    for idx in range(len(FASTING_BLOOD_SUGAR_TYPES)):
        df.loc[df['fasting_blood_sugar'] == idx, 'fasting_blood_sugar'] = FASTING_BLOOD_SUGAR_TYPES[idx]

    for idx in range(len(REST_ECG_TYPES)):
        df.loc[df['rest_ecg'] == idx, 'rest_ecg'] = REST_ECG_TYPES[idx]

    for idx in range(len(EXERCISE_INDUCED_ANGINA_TYPES)):
        df.loc[df['exercise_induced_angina'] == idx, 'exercise_induced_angina'] = EXERCISE_INDUCED_ANGINA_TYPES[idx]

    for idx in range(len(ST_SLOPE_TYPES)):
        df.loc[df['st_slope'] == idx, 'st_slope'] = ST_SLOPE_TYPES[idx]

    for idx in range(len(THALASSEMIA_TYPES)):
        df.loc[df['thalassemia'] == idx, 'thalassemia'] = THALASSEMIA_TYPES[idx]

    return df, FINAL_CAT_COLUMNS, FINAL_NUM_COLUMNS


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