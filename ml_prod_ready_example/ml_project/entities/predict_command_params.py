import yaml
from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
from omegaconf import OmegaConf, DictConfig


from .feature_params import FeatureParams


@dataclass()
class PredictCommandParams:
    input_data_path: str = field(default='')
    trained_model_path: str = field(default='')
    output_preds_path: str = field(default='')
    feature_params: FeatureParams = field(default=FeatureParams())


PredictCommandParamsSchema = class_schema(PredictCommandParams)


def read_predict_command_params(raw_cfg: DictConfig) -> PredictCommandParamsSchema:
    schema = PredictCommandParamsSchema()
    cfg = OmegaConf.to_yaml(raw_cfg)
    return schema.load(yaml.safe_load(cfg))
