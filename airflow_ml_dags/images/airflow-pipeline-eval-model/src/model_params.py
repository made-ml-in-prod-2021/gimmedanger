from dataclasses import dataclass

@dataclass()
class ModelParams:
    random_state: int
    n_estimators: int
    max_depth: int
