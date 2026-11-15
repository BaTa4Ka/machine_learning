# Notebooks

These notebooks document the project workflow from data setup and auditing through
preprocessing, modelling, comparison, and analysis.

## Data preparation and auditing

- [`setup.ipynb`](setup.ipynb) — downloads the monthly parquet files required by the
  project.
- [`data_audit.ipynb`](data_audit.ipynb) — inspects dataset shape, column types, missing
  values, duplicates, target distribution, and possible outliers.
- [`one_hot_features_audit.ipynb`](one_hot_features_audit.ipynb) — examines categorical
  feature cardinality and supports decisions about rare-category grouping and one-hot
  encoding.
- [`preprocessing.ipynb`](preprocessing.ipynb) — demonstrates target creation,
  chronological splitting, filtering, feature derivation, and removal of technical
  columns.

## Baselines and modelling

- [`baseline.ipynb`](baseline.ipynb) — evaluates majority-class baseline performance
  by train type using accuracy and macro-F1.
- [`linear_modeling.ipynb`](linear_modeling.ipynb) — trains and selects the logistic
  regression reference model using validation macro-F1.
- [`boosted_modeling.ipynb`](boosted_modeling.ipynb) — compares exploratory boosted
  tree models, including XGBoost, LightGBM, and CatBoost.
- [`model_tuning.ipynb`](model_tuning.ipynb) — tunes the logistic regression and
  LightGBM configurations using the chronological validation partition.
- [`model_comparison.ipynb`](model_comparison.ipynb) — compares the majority baseline,
  logistic regression, and LightGBM; reports per-class metrics, train-type metrics,
  confusion matrices, and the final LightGBM test evaluation.

## Exploratory alternatives

- [`bagging_vs_random_forest.ipynb`](bagging_vs_random_forest.ipynb) — compares bagging
  and random forest models and documents why they are not retained as main models.
- [`neural_network_modeling.ipynb`](neural_network_modeling.ipynb) — evaluates an
  sklearn multilayer perceptron with ReLU hidden layers and a softmax output.