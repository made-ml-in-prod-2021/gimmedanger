#!/bin/bash

printf '\nTesting prepare_dataset...\n'
python -m unittest tests/data/test_prepare_dataset.py

printf '\nTesting build transformer...\n'
python -m unittest tests/features/test_build_transformer.py

printf '\nTesting grid search...\n'
python -m unittest tests/grid_search/test_grid_search.py

printf '\nTesting model fit predict...\n'
python -m unittest tests/model/test_model_fit_predict.py
