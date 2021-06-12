from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago


LOCAL_CONFIG_PATH = 'configs/generate_data.yaml'
LOCAL_OUTPUT_PATH = 'data/raw/{{ ds }}/data.csv'
PROJECT_PATH = '/opt/hhd-drive/Yandex.Disk/Computer Science/courses/ml/made-ml-in-prod/hws/airflow_ml_dags'


with DAG(
    dag_id='generate-data',
    start_date=days_ago(1),
    schedule_interval='@hourly',
    max_active_runs=1,
) as dag:

    generate = DockerOperator(
        command=f'{LOCAL_CONFIG_PATH} {LOCAL_OUTPUT_PATH}',
        image='airflow-generate-data',
        network_mode='bridge',
        task_id='docker-airflow-generate-data',
        do_xcom_push=False,
        volumes=[f'{PROJECT_PATH}/configs:/configs', f'{PROJECT_PATH}/data:/data']
    )
