import yaml
from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
from omegaconf import OmegaConf, DictConfig


@dataclass()
class SplitTrainValParams:
    test_size: float = field(default=0.2)
    random_state: int = field(default=13)


SplitTrainValParamsSchema = class_schema(SplitTrainValParams)


def read_split_train_val_params(raw_cfg: DictConfig) -> SplitTrainValParamsSchema:
    schema = SplitTrainValParamsSchema()
    cfg = OmegaConf.to_yaml(raw_cfg)
    return schema.load(yaml.safe_load(cfg))
