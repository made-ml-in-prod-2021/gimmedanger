FROM airflow-ml-base:latest

COPY setup.py /setup.py
COPY prepare_command.py /prepare_command.py

WORKDIR .

RUN ["pip", "install", "-e", "."]
ENTRYPOINT ["python", "prepare_command.py"]
