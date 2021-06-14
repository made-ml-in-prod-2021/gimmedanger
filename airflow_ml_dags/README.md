# Homework 3 Airflow Data Pipeline

# Airflow data pipelines for scheduled generation, training and interference 

## Local run:

Installation:
~~~
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -e .
~~~

## Run Airflow UI:

Installation:
~~~
$ docker-compose up --build
~~~

Now open http://0.0.0.0:8080 in browser and follow UI, you will see generate_data, training_pipeline and predict dags

