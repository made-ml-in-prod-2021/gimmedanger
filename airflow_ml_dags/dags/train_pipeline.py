from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago


LOCAL_INPUT_PATH = 'data/raw/{{ ds }}/data.csv'
LOCAL_OUTPUT_PATH = 'data/processed/{{ ds }}/train_data.csv'

LOCAL_CONFIG_PATH = 'configs/generate_data.yaml'
PROJECT_PATH = '/opt/hhd-drive/Yandex.Disk/Computer Science/courses/ml/made-ml-in-prod/hws/airflow_ml_dags'


with DAG(
    dag_id='training-pipeline',
    start_date=days_ago(1),
    schedule_interval='@weekly',
    max_active_runs=1,
) as dag:

    prepare_data = DockerOperator(
        command=f'{LOCAL_INPUT_PATH} {LOCAL_OUTPUT_PATH}',
        image='airflow-pipeline-prepare-data',
        task_id='docker-airflow-pipeline-prepare-data',
        do_xcom_push=False,
        network_mode='bridge',
        volumes=[f'{PROJECT_PATH}/configs:/configs', f'{PROJECT_PATH}/data:/data']
    )

    train_val_split = BashOperator(task_id="1", bash_command='echo "data spliting!!!"')
    train_model = BashOperator(task_id="2", bash_command='echo "model training!!!"')
    eval_model = BashOperator(task_id="3", bash_command='echo "model evaluationg!!!"')

    prepare_data >> train_val_split >> train_model >> eval_model