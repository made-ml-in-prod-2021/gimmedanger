from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from constants import PROJECT_PATH, DEFAULT_ARGS


with DAG(
    dag_id='predict',
    default_args=DEFAULT_ARGS,
    start_date=days_ago(1),
    schedule_interval='@daily',
    max_active_runs=1,
) as dag:

    PROD_MODEL_NAME = 'model.pkl'
    PREDICT_MODEL_PATH = f'data/models/{PROD_MODEL_NAME}'
    PREDICT_INPUT_PATH = 'data/raw/{{ ds }}/predict_data.csv'
    PREDICT_OUTPUT_PATH = 'data/preds/{{ ds }}.csv'
    predict_model_sensor = FileSensor(
        task_id='predict_model_sensor',
        filepath=PREDICT_MODEL_PATH
    )
    predict_input_data_sensor = FileSensor(
        task_id='predict_input_data_sensor',
        filepath=PREDICT_INPUT_PATH
    )
    predict_model = DockerOperator(
        command=f'{PREDICT_MODEL_PATH} {PREDICT_INPUT_PATH} {PREDICT_OUTPUT_PATH}',
        image='airflow-predict',
        task_id='docker-airflow-predict',
        do_xcom_push=False,
        network_mode='bridge',
        volumes=[f'{PROJECT_PATH}/configs:/configs', f'{PROJECT_PATH}/data:/data']
    )

    predict_model_sensor >> predict_input_data_sensor >> predict_model
