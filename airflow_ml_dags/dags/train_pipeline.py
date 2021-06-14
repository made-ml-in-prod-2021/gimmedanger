from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago


PROJECT_PATH = '/opt/hhd-drive/Yandex.Disk/Computer Science/courses/ml/made-ml-in-prod/hws/airflow_ml_dags'


with DAG(
    dag_id='training-pipeline',
    start_date=days_ago(1),
    schedule_interval='@weekly',
    max_active_runs=1,
) as dag:

    PREPARE_INPUT_PATH = 'data/raw/{{ ds }}/data.csv'
    PREPARE_OUTPUT_PATH = 'data/processed/{{ ds }}/data.csv'
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
    eval_model = DockerOperator(
        command=f'{TRAIN_CONFIG_PATH} {EVAL_MODEL_PATH} {EVAL_INPUT_PATH} {EVAL_OUTPUT_PATH}',
        image='airflow-pipeline-eval-model',
        task_id='docker-airflow-pipeline-eval-model',
        do_xcom_push=False,
        network_mode='bridge',
        volumes=[f'{PROJECT_PATH}/configs:/configs', f'{PROJECT_PATH}/data:/data']
    )

    prepare_data >> train_val_split >> train_model >> eval_model