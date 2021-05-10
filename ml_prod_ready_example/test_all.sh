#!/bin/bash

printf '\nTesting prepare_dataset...\n'
python -m unittest tests/data/test_prepare_dataset.py

printf '\nTesting build transformer...\n'
python -m unittest tests/features/test_build_transformer.py
