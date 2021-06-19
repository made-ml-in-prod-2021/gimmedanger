from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from constants import PROJECT_PATH, DEFAULT_ARGS


with DAG(
    dag_id='generate-data',
    default_args=DEFAULT_ARGS,
    start_date=days_ago(1),
    schedule_interval='@hourly',
    max_active_runs=1,
) as dag:

    LOCAL_CONFIG_PATH = 'configs/generate_data.yaml'
    LOCAL_OUTPUT_PATH = 'data/raw/{{ ds }}/data.csv'
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
