import yaml
from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
from omegaconf import OmegaConf, DictConfig

from .feature_params import FeatureParams


@dataclass()
class GenerateDataParams:
    dataset_rows: int
    target_required: bool
    path_historical_data: str
    feature_params: FeatureParams = field(default=FeatureParams())
    random_seed: int = field(default=0)


GenerateDataParamsSchema = class_schema(GenerateDataParams)


def read_generate_data_params(raw_cfg: DictConfig) -> GenerateDataParamsSchema:
    schema = GenerateDataParamsSchema()
    cfg = OmegaConf.to_yaml(raw_cfg)
    return schema.load(yaml.safe_load(cfg))
