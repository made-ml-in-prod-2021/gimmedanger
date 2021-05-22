from dataclasses import dataclass, field
from src.data import FINAL_CAT_COLUMNS, FINAL_NUM_COLUMNS, TARGET_COLUMN
from typing import List


@dataclass()
class FeatureParams:
    target_col: str = field(default=TARGET_COLUMN)
    categorical_features: List[str] = field(default_factory=lambda: list(FINAL_CAT_COLUMNS))
    numerical_features: List[str] = field(default_factory=lambda: list(FINAL_NUM_COLUMNS))
