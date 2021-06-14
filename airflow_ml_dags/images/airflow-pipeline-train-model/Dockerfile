FROM airflow-ml-base:latest

COPY src /src
COPY tests /tests

COPY setup.py /setup.py
COPY requirements.txt /requirements.txt
COPY train_command.py /train_command.py

WORKDIR .

RUN ["pip", "install", "-e", "."]
RUN ["python", "-m", "unittest", "tests/test_build_transformer.py"]
RUN ["python", "-m", "unittest", "tests/test_model_fit_predict.py"]
ENTRYPOINT ["python", "train_command.py"]
