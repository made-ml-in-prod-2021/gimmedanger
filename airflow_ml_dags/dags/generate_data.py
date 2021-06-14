from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago


LOCAL_CONFIG_PATH = 'configs/generate_data.yaml'
LOCAL_OUTPUT_PATH = 'data/raw/{{ ds }}/data.csv'
PROJECT_PATH = '/opt/hhd-drive/Yandex.Disk/Computer Science/courses/ml/made-ml-in-prod/hws/airflow_ml_dags'


default_args = {
    "owner": "airflow",
    "email": ["vladim1r.nazarov@yandex.ru"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id='generate-data',
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval='@hourly',
    max_active_runs=1,
) as dag:
    config_sensor = FileSensor(
        task_id='config_sensor',
        filepath=LOCAL_CONFIG_PATH
    )
    generate = DockerOperator(
        command=f'{LOCAL_CONFIG_PATH} {LOCAL_OUTPUT_PATH}',
        image='airflow-generate-data',
        network_mode='bridge',
        task_id='docker-airflow-generate-data',
        do_xcom_push=False,
        volumes=[f'{PROJECT_PATH}/configs:/configs', f'{PROJECT_PATH}/data:/data']
    )

    config_sensor >> generate
