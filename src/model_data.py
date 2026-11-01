"""Prepare encoded train, validation, and test data for model training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .prep_job import (
        add_split_time,
        add_target,
        chronological_split,
        load_monthly_data,
        preprocess_partitions,
    )
    from .sklearn_preprocessing import (
        make_feature_preprocessor,
        prepare_model_features,
    )
except ImportError:
    from prep_job import (
        add_split_time,
        add_target,
        chronological_split,
        load_monthly_data,
        preprocess_partitions,
    )
    from sklearn_preprocessing import (
        make_feature_preprocessor,
        prepare_model_features,
    )

RAW_MODEL_COLUMNS = (
    "arrival_planned_time",
    "departure_planned_time",
    "time",
    "delay_in_min",
    "arrival_is_canceled",
    "train_type",
    "line_number",
    "station_name",
    "train_number",
    "train_line_ride_id",
    "train_line_station_num",
    "final_destination_station",
)


# Keep encoded matrices, labels, and the fitted transformer together.
@dataclass
class ModelData:
    """Encoded partitions and the fitted feature transformer."""

    X_train: Any
    y_train: pd.Series
    X_validation: Any
    y_validation: pd.Series
    X_test: Any
    y_test: pd.Series
    preprocessor: object


# Prepare all model partitions through the shared project workflow.
def prepare_model_data(data_dir: Path | str) -> ModelData:
    """Run project preprocessing and fit feature transformations on training data."""
    # Run dataset-specific loading, labelling, and chronological splitting first.
    raw_data = load_monthly_data(data_dir, RAW_MODEL_COLUMNS)
    labelled_data = add_target(raw_data)
    split_data = chronological_split(add_split_time(labelled_data))
    processed_data = preprocess_partitions(split_data)

    # Separate model inputs from labels for each partition.
    train_features, y_train = prepare_model_features(processed_data["train"])
    validation_features, y_validation = prepare_model_features(
        processed_data["validation"]
    )
    test_features, y_test = prepare_model_features(processed_data["test"])

    # Fit feature rules on training data only to prevent temporal leakage.
    preprocessor = make_feature_preprocessor()
    X_train = preprocessor.fit_transform(train_features)
    X_validation = preprocessor.transform(validation_features)
    X_test = preprocessor.transform(test_features)

    return ModelData(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        preprocessor=preprocessor,
    )
