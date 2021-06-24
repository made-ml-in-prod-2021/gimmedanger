from datetime import timedelta

PROJECT_PATH = '/opt/hhd-drive/Yandex.Disk/Computer Science/courses/ml/made-ml-in-prod/hws/airflow_ml_dags'

DEFAULT_ARGS = {
    "owner": "airflow",
    "email": ["vladim1r.nazarov@yandex.ru"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
