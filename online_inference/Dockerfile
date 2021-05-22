FROM python:3.6

# Sources and interference input
COPY models/model.pkl /models/model.pkl
COPY configs/predict_params.yaml /configs/predict_params.yaml
COPY src /src

# Scripts
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py
COPY app.py /app.py

WORKDIR .

RUN ["pip", "install", "-e", "."]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
