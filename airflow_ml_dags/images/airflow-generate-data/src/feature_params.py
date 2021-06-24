from dataclasses import dataclass, field
from typing import List

from .dataset_defaults import FINAL_CAT_COLUMNS, FINAL_NUM_COLUMNS, TARGET_COLUMN


@dataclass()
class FeatureParams:
    target_col: str = field(default=TARGET_COLUMN)
    categorical_features: List[str] = field(default_factory=lambda: list(FINAL_CAT_COLUMNS))
    numerical_features: List[str] = field(default_factory=lambda: list(FINAL_NUM_COLUMNS))
