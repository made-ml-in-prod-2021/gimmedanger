from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago


PROJECT_PATH = '/opt/hhd-drive/Yandex.Disk/Computer Science/courses/ml/made-ml-in-prod/hws/airflow_ml_dags'

default_args = {
    "owner": "airflow",
    "email": ["vladim1r.nazarov@yandex.ru"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id='training-pipeline',
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval='@weekly',
    max_active_runs=1,
) as dag:

    PREPARE_INPUT_PATH = 'data/raw/{{ ds }}/data.csv'
    PREPARE_OUTPUT_PATH = 'data/processed/{{ ds }}/data.csv'
    prepare_input_data_sensor = FileSensor(
        task_id='prepare_input_data_sensor',
        filepath=PREPARE_INPUT_PATH
    )
    prepare_data = DockerOperator(
        command=f'{PREPARE_INPUT_PATH} {PREPARE_OUTPUT_PATH}',
        image='airflow-pipeline-prepare-data',
        task_id='docker-airflow-pipeline-prepare-data',
        do_xcom_push=False,
        network_mode='bridge',
        volumes=[f'{PROJECT_PATH}/configs:/configs', f'{PROJECT_PATH}/data:/data']
    )

    SPLIT_CONFIG_PATH = 'configs/train_val_split.yaml'
    SPLIT_INPUT_PATH = 'data/processed/{{ ds }}/data.csv'
    SPLIT_OUTPUT_PATH = 'data/processed/{{ ds }}'
    split_config_sensor = FileSensor(
        task_id='split_config_sensor',
        filepath=SPLIT_CONFIG_PATH
    )
    split_input_data_sensor = FileSensor(
        task_id='split_input_data_sensor',
        filepath=SPLIT_INPUT_PATH
    )
    train_val_split = DockerOperator(
        command=f'{SPLIT_CONFIG_PATH} {SPLIT_INPUT_PATH} {SPLIT_OUTPUT_PATH}',
        image='airflow-pipeline-train-val-split',
        task_id='docker-airflow-pipeline-train-val-split',
        do_xcom_push=False,
        network_mode='bridge',
        volumes=[f'{PROJECT_PATH}/configs:/configs', f'{PROJECT_PATH}/data:/data']
    )

    TRAIN_CONFIG_PATH = 'configs/train_model.yaml'
    TRAIN_INPUT_PATH = 'data/processed/{{ ds }}/train_data.csv'
    TRAIN_OUTPUT_PATH = 'data/models/{{ ds }}.pkl'
    train_config_sensor = FileSensor(
        task_id='train_config_sensor',
        filepath=TRAIN_CONFIG_PATH
    )
    train_input_data_sensor = FileSensor(
        task_id='train_input_data_sensor',
        filepath=TRAIN_INPUT_PATH
    )
    train_model = DockerOperator(
        command=f'{TRAIN_CONFIG_PATH} {TRAIN_INPUT_PATH} {TRAIN_OUTPUT_PATH}',
        image='airflow-pipeline-train-model',
        task_id='docker-airflow-pipeline-train-model',
        do_xcom_push=False,
        network_mode='bridge',
        volumes=[f'{PROJECT_PATH}/configs:/configs', f'{PROJECT_PATH}/data:/data']
    )

    EVAL_MODEL_PATH = 'data/models/{{ ds }}.pkl'
    EVAL_INPUT_PATH = 'data/processed/{{ ds }}/val_data.csv'
    EVAL_OUTPUT_PATH = 'data/evals/{{ ds }}.txt'
    eval_model_sensor = FileSensor(
        task_id='eval_model_sensor',
        filepath=EVAL_MODEL_PATH
    )
    eval_input_data_sensor = FileSensor(
        task_id='eval_input_data_sensor',
        filepath=EVAL_INPUT_PATH
    )
    eval_model = DockerOperator(
        command=f'{TRAIN_CONFIG_PATH} {EVAL_MODEL_PATH} {EVAL_INPUT_PATH} {EVAL_OUTPUT_PATH}',
        image='airflow-pipeline-eval-model',
        task_id='docker-airflow-pipeline-eval-model',
        do_xcom_push=False,
        network_mode='bridge',
        volumes=[f'{PROJECT_PATH}/configs:/configs', f'{PROJECT_PATH}/data:/data']
    )

    prepare_input_data_sensor >> prepare_data >> \
    split_config_sensor >> split_input_data_sensor >> train_val_split >> \
    train_config_sensor >> train_input_data_sensor >> train_model >> \
    eval_model_sensor >> eval_input_data_sensor >> eval_model