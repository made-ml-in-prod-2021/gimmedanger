from dataclasses import dataclass, field


@dataclass()
class SplitTrainValParams:
    test_size: float = field(default=0.2)
    random_state: int = field(default=13)