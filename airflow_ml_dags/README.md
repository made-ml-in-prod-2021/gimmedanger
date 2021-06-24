# Homework 3 Airflow Data Pipeline

# Airflow data pipelines for scheduled generation, training and interference 

## Local run:

Installation:
~~~
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -e .
~~~

Now set `PROJECT_PATH` with the full system path to this folder in [dags/constants.py](https://github.com/made-ml-in-prod-2021/gimmedanger/tree/homework3/airflow_ml_dags/dags/constants.py)

## Run Airflow UI:

Installation:
~~~
$ docker-compose up --build
~~~

Now open http://0.0.0.0:8080 in browser and follow UI, you will see generate_data, training_pipeline and predict dags

## Required parameters in UI:

You need to create `fs_default` file connection. Go to `Admin -> Connections` and add new record: conn_id=fs_default and conn_type=file, other fields should be default