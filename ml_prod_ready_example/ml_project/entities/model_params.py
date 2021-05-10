from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier
from typing import List

@dataclass()
class ModelParams:
    model_type: object = field(default=RandomForestClassifier)
    random_state: int = field(default=9050)
    n_estimators: int = field(default=100)
    max_depth: int = field(default=5)


@dataclass()
class SearchParams:
    n_estimator_lst: List[int] = field(default_factory=lambda: [50, 100, 200])
    max_depth_lst: List[int] = field(default_factory=lambda: [5, 7, 9, 11, 13])
