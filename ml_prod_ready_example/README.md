Homework 1 ML production ready example
==============================

Installation: 
~~~
$ virtualenv -p python3.8 .venv
$ source .venv/bin/activate
$ pip install -e .
~~~
Train with grid search for parameters:
~~~
$ python ml_project/train_pipeline.py --config-name grid_search_train
~~~

Train with test/val split for custom parameters:
~~~
$ python ml_project/train_pipeline.py --config-name best_params_train
~~~

Predict with pretrained model:
~~~
$ python ml_project/predict.py --config-name predict_params.yaml
~~~

Test:
~~~
$ bash test_all.sh
~~~

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── ml_project         <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── train_pipeline.py  <- Full pipelines for training (with test/val split or grid search cv)
    │   │
    │   ├── data               <- download or generate data
    │   │
    │   ├── emtities           <- parameters dataclasses
    │   │
    │   ├── features           <- turn raw data into features for modeling
    │   │
    │   ├── grid_search        <- grid search train for to find best model
    │   │
    │   └── models             <- train models and then use trained models to make
    │
    ├── test_all.sh        <- Script for testing
    └── tests              <- Tests for all modules 

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
