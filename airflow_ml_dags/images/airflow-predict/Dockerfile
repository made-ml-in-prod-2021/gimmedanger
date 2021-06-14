FROM airflow-ml-base:latest

COPY src /src

COPY setup.py /setup.py
COPY requirements.txt /requirements.txt
COPY predict_command.py /predict_command.py

WORKDIR .

RUN ["pip", "install", "-e", "."]
ENTRYPOINT ["python", "predict_command.py"]
