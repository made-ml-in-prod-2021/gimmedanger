# Homework 2 Docker and REST API

REST-service implementation for interference with ML model implemented in [hw1](https://github.com/made-ml-in-prod-2021/gimmedanger/tree/main/ml_prod_ready_example)

## Local run:

Installation and sanity check:
~~~
$ virtualenv -p python3.8 .venv
$ source .venv/bin/activate
$ pip install -e .
$ pytest
~~~

FastAPI server run:
~~~
$ uvicorn app:app
~~~

Make request for interference:
~~~
$ python make_request.py http://127.0.0.1:8000/ data/test_predict_dataset.csv result.json
~~~

## Docker run:

Build docker image:
~~~
$ sudo docker build -t gimmedanger/made-ml-in-prod-hw2:v1 .
~~~

Run docker image:
~~~
$ docker run -p 80:80 gimmedanger/made-ml-in-prod-hw2:v1
~~~

Now the app is running on local server 'http://localhost/' and we could send prediction requests:
~~~
$ python make_request.py http://localhost/ data/test_predict_dataset.csv result.json
~~~

For this test input server logs the following:
~~~
2021-05-22 12:13:19,038 - app - INFO - Data validation started
2021-05-22 12:13:19,038 - app - INFO - Dataframe construction started
2021-05-22 12:13:19,055 - app - INFO - Dataframe shape is (50, 13)
2021-05-22 12:13:19,055 - app - INFO - Pipeline started
2021-05-22 12:13:19,072 - app - INFO - Predictions shape is (50,)
INFO:     172.17.0.1:57748 - "GET /predict/ HTTP/1.1" 200 OK
~~~

Built docker image was pushed to [hub.docker](https://hub.docker.com/repository/docker/gimmedanger/made-ml-in-prod-hw2) and could be used:
~~~
$ docker pull gimmedanger/made-ml-in-prod-hw2:v1
$ docker run -p 80:80 gimmedanger/made-ml-in-prod-hw2:v1
~~~
