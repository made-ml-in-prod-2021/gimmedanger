from dataclasses import dataclass
from typing import List


@dataclass()
class FeatureParams:
    target_col: str
    categorical_features: List[str]
    numerical_features: List[str]
