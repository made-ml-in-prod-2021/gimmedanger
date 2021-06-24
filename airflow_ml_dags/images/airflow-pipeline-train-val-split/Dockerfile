FROM airflow-ml-base:latest

COPY src /src
COPY tests /tests
COPY setup.py /setup.py
COPY requirements.txt /requirements.txt
COPY split_command.py /split_command.py

WORKDIR .

RUN ["pip", "install", "-e", "."]
RUN ["python", "-m", "unittest", "tests/test_split_train_val.py"]

ENTRYPOINT ["python", "split_command.py"]
