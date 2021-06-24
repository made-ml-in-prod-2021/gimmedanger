FROM airflow-ml-base:latest

COPY src /src
COPY tests /tests

COPY setup.py /setup.py
COPY requirements.txt /requirements.txt
COPY generate_command.py /generate_command.py

WORKDIR .

RUN ["pip", "install", "-e", "."]
RUN ["python", "-m", "unittest", "tests/test_prepare_dataset.py"]
ENTRYPOINT ["python", "generate_command.py"]
