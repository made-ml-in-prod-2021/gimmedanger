import pandas as pd
import numpy as np

from .generate_data_params import GenerateDataParams
from .dataset_defaults \
    import ORIG_COLUMNS, THALASSEMIA_TYPES, \
    SEX_TYPES, CHEST_PAIN_TYPES, FASTING_BLOOD_SUGAR_TYPES, \
    REST_ECG_TYPES, EXERCISE_INDUCED_ANGINA_TYPES, ST_SLOPE_TYPES, \
    FINAL_COLUMNS


def read_data(path: str) -> pd.DataFrame:
    """
    read_csv wrapper with asserts
    :param path: dataset path
    :return: loaded dataframe
    """
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: original dataframe
    :return: (modified dataframe, categorical features names, numerical features names)
    """
    df = df.rename(columns=dict(zip(ORIG_COLUMNS, FINAL_COLUMNS)))

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

    return df


def generate_data(params: GenerateDataParams):
    """
    :param params: generation params
    :return: synthetic dataframe
    """
    np.random.seed(params.random_seed)
    feature_params = params.feature_params
    cat_features = feature_params.categorical_features
    num_features = feature_params.numerical_features
    columns = cat_features + num_features + (['target'] if params.target_required else [])
    df = pd.DataFrame(None, index=np.arange(params.dataset_rows), columns=columns)
    hist_df = prepare_data(read_data(params.path_historical_data))
    for f in num_features:
        df[f] = np.random.uniform(low=hist_df[f].min(), high=hist_df[f].max(), size=params.dataset_rows)
    for f in cat_features:
        categories = None
        if f == 'sex':
            categories = SEX_TYPES
        elif f == 'chest_pain_type':
            categories = CHEST_PAIN_TYPES
        elif f == 'fasting_blood_sugar':
            categories = FASTING_BLOOD_SUGAR_TYPES
        elif f == 'rest_ecg':
            categories = REST_ECG_TYPES
        elif f == 'exercise_induced_angina':
            categories = EXERCISE_INDUCED_ANGINA_TYPES
        elif f == 'st_slope':
            categories = ST_SLOPE_TYPES
        elif f == 'thalassemia':
            categories = THALASSEMIA_TYPES
        df[f] = [categories[idx] for idx in np.random.randint(len(categories), size=params.dataset_rows)]
    if params.target_required:
        df[feature_params.target_col] = np.random.randint(1, params.dataset_rows)
    return df
