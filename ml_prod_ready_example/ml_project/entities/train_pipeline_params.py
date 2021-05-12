import yaml
from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
from omegaconf import OmegaConf, DictConfig

from .model_params import ModelParams, SearchParams
from .feature_params import FeatureParams
from .split_train_val_params import SplitTrainValParams


@dataclass()
class TrainingPipelineParams:
    experiment_name: str = field(default='')
    input_data_path: str = field(default='')
    output_model_path: str = field(default='')
    run_grid_search: bool = field(default=False)
    split_train_val_params: SplitTrainValParams = field(default=SplitTrainValParams())
    feature_params: FeatureParams = field(default=FeatureParams())
    model_params: ModelParams = field(default=ModelParams())
    search_params: SearchParams = field(default=SearchParams())


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(raw_cfg: DictConfig) -> TrainingPipelineParams:
    schema = TrainingPipelineParamsSchema()
    cfg = OmegaConf.to_yaml(raw_cfg)
    return schema.load(yaml.safe_load(cfg))
