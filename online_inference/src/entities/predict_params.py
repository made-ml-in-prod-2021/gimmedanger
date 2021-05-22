import yaml
from dataclasses import dataclass
from marshmallow_dataclass import class_schema

from .feature_params import FeatureParams


@dataclass()
class PredictParams:
    trained_model_path: str
    output_preds_path: str
    feature_params: FeatureParams


PredictParamsSchema = class_schema(PredictParams)


def read_predict_params(path: str) -> PredictParams:
    schema = PredictParamsSchema()
    with open(path, "r") as input_stream:
        return schema.load(yaml.safe_load(input_stream))
