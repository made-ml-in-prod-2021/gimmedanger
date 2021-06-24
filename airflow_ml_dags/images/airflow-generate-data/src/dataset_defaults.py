TARGET_COLUMN = 'target'

ORIG_SHAPE = (303, 14)

ORIG_COLUMNS = ('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target')

FINAL_COLUMNS = ('age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
                 'cholesterol', 'fasting_blood_sugar', 'rest_ecg',
                 'max_heart_rate_achieved', 'exercise_induced_angina',
                 'st_depression', 'st_slope', 'num_major_vessels',
                 'thalassemia', 'target')

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

FINAL_CAT_COLUMNS = ('sex', 'chest_pain_type', 'fasting_blood_sugar',
                     'rest_ecg', 'exercise_induced_angina',
                     'st_slope', 'thalassemia')

FINAL_NUM_COLUMNS = tuple(set(FINAL_COLUMNS) - set(FINAL_CAT_COLUMNS) - {'target'})
