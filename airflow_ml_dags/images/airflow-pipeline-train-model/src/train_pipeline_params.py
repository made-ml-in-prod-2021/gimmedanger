import yaml
from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
from omegaconf import OmegaConf, DictConfig

from .model_params import ModelParams
from .feature_params import FeatureParams


@dataclass()
class TrainingPipelineParams:
    experiment_name: str
    feature_params: FeatureParams
    model_params: ModelParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(raw_cfg: DictConfig) -> TrainingPipelineParams:
    schema = TrainingPipelineParamsSchema()
    cfg = OmegaConf.to_yaml(raw_cfg)
    return schema.load(yaml.safe_load(cfg))
