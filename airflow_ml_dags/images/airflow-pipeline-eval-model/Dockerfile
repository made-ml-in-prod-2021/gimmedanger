FROM airflow-ml-base:latest

COPY src /src

COPY setup.py /setup.py
COPY requirements.txt /requirements.txt
COPY eval_command.py /eval_command.py

WORKDIR .

RUN ["pip", "install", "-e", "."]
ENTRYPOINT ["python", "eval_command.py"]
