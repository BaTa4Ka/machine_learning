# Source code

This folder contains reusable data-loading, preprocessing, modelling, and evaluation
utilities used by the notebooks.

- [`data_utils.py`](data_utils.py) — shared constants, monthly-file discovery, data
  loading, and target-construction helpers.
- [`download_data.py`](download_data.py) — downloads and validates the monthly source
  parquet files.
- [`prep_job.py`](prep_job.py) — runs the leakage-safe preparation workflow, including
  planned-time derivation, chronological splitting, target creation, filtering, and
  feature derivation.
- [`sklearn_preprocessing.py`](sklearn_preprocessing.py) — defines the sparse
  scikit-learn preprocessing pipeline for scaling numeric features, grouping rare
  categories, handling missing values, and one-hot encoding.
- [`boosted_preprocessing.py`](boosted_preprocessing.py) — prepares categorical
  features for boosted tree models without one-hot encoding.
- [`model_data.py`](model_data.py) — combines the preparation and sklearn preprocessing
  steps and returns train, validation, and test model data.
- [`baselines.py`](baselines.py) — contains simple fitted baseline classifiers,
  including the majority-class classifier.
- [`models.py`](models.py) — constructs the logistic regression model used as the
  linear reference.
- [`model_evaluation.py`](model_evaluation.py) — provides accuracy and macro-F1
  evaluation helpers for fitted models.