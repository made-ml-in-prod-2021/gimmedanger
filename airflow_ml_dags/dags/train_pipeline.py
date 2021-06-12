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
    train_model = BashOperator(task_id="2", bash_command='echo "model training!!!"')
    eval_model = BashOperator(task_id="3", bash_command='echo "model evaluationg!!!"')

    prepare_data >> train_val_split >> train_model >> eval_model