import sys
import pickle
import logging
import uvicorn

import pandas as pd
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from fastapi import FastAPI, HTTPException
from typing import List, Union, Optional, NoReturn

from src.data import prepare_data
from src.entities import read_predict_params, PredictParams


logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


class XInput(BaseModel):
    data: List[List[Union[float, str]]]
    features: List[str]


class YResponse(BaseModel):
    predict: int


def deserialize_model(path: str) -> Pipeline:
    with open(path, 'rb') as f:
        return pickle.load(f)


def check_data(
        data: List[List[Union[float, str]]],
        features: List[str],
        params: PredictParams
) -> NoReturn:
    input_features_set = set(features)
    feature_params = params.feature_params
    expected_features = feature_params.numerical_features + feature_params.categorical_features

    if not len(expected_features) == len(features):
        raise HTTPException(status_code=400, detail='Wrong number of features')

    for name in expected_features:
        if name not in input_features_set:
            raise HTTPException(status_code=400, detail='Wrong names of features')

    for x in data:
        if not len(x) == len(features):
            raise HTTPException(status_code=400, detail='Wrong number of features in data')


def predict_model(
        data: List[List[Union[float, str]]],
        features: List[str],
        pipe: Pipeline,
        params: PredictParams
) -> List[YResponse]:
    logger.info(f'Data validation started')
    check_data(data, features, params)
    logger.info(f'Dataframe construction started')
    df = prepare_data(pd.DataFrame(data, columns=features))
    logger.info(f'Dataframe shape is {df.shape}')
    logger.info(f'Pipeline started')
    preds = pipe.predict(df)
    logger.info(f'Predictions shape is {preds.shape}')
    return [YResponse(predict=p) for p in list(preds)]


app = FastAPI()
model: Optional[Pipeline] = None
predict_params: Optional[PredictParams] = None


@app.get('/')
async def main():
    return 'Entry point of prediction model'


@app.on_event('startup')
def load_model():
    global model
    global predict_params
    logger.info('App initialization started')
    predict_params = read_predict_params(path=f'configs/predict_params.yaml')
    logger.info(f'Loaded config: {predict_params}')
    model = deserialize_model(predict_params.trained_model_path)
    logger.info(f'Loaded model: {model}')
    logger.info('App initialization finished')


@app.get('/predict/', response_model=List[YResponse])
async def predict(request: XInput):
    return predict_model(request.data, request.features, model, predict_params)


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=80)
