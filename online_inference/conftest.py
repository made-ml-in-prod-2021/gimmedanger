import os
import pytest


@pytest.fixture()
def test_predict_dataset_path():
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, "data/test_predict_dataset.csv")
