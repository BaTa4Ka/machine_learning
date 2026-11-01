"""Scikit-learn feature preprocessing for the delay models."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

TARGET_COLUMN = "target"
# This ordered numeric feature is scaled separately from categorical inputs.
NUMERIC_FEATURES = ("train_line_station_num",)
# These nominal values are represented as sparse one-hot indicator columns.
CATEGORICAL_FEATURES = (
    "train_type",
    "day_of_week",
    "planned_hour",
    "first_station",
    "station_name",
    "final_destination_station",
    "train_number",
)


class AverageCategoryFrequencyGrouper(BaseEstimator, TransformerMixin):
    """Group categories that occur less often than their feature average."""

    # Store labels used for infrequent and missing categorical values.
    def __init__(
        self,
        *,
        rare_value: str = "__rare__",
        missing_value: str = "__missing__",
    ) -> None:
        self.rare_value = rare_value
        self.missing_value = missing_value

    # Learn category frequencies and the accepted categories from training data.
    def fit(self, X: pd.DataFrame, y: object = None):
        # Keep the input as a DataFrame so feature names remain available.
        frame = self._validate_input(X)
        self.feature_names_in_ = list(frame.columns)
        # Store the training-derived rules reused by validation and test data.
        self.common_categories_: dict[str, set[object]] = {}
        self.average_category_counts_: dict[str, float] = {}

        # Calculate a separate average and category list for every feature.
        for column in self.feature_names_in_:
            # Count how often each category appears in this feature.
            category_counts = (
                frame[column]
                .astype("string")
                .fillna(self.missing_value)
                .value_counts()
            )
            # Use the mean category size as the minimum frequency threshold.
            self.average_category_counts_[column] = (
                len(frame) / len(category_counts)
            )
            # Keep categories meeting the threshold; the rest become __rare__.
            self.common_categories_[column] = set(
                category_counts[
                    category_counts >= self.average_category_counts_[column]
                ].index
            )
        return self

    # Apply the training-derived category mapping to later data partitions.
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Copy the selected columns so the caller's DataFrame is not modified.
        frame = self._validate_input(X).copy()
        # Apply the same training-derived category rules to every feature.
        for column in self.feature_names_in_:
            # Convert values to one consistent string representation.
            values = frame[column].astype("string").fillna(self.missing_value)
            frame[column] = values.where(
                values.isin(self.common_categories_[column]),
                self.rare_value,
            )
        return frame

    # Validate the input shape and return the categorical columns in a fixed order.
    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("categorical features must be provided as a DataFrame")
        missing = set(CATEGORICAL_FEATURES) - set(X.columns)
        if missing:
            raise KeyError(f"Missing categorical features: {sorted(missing)}")
        return X[list(CATEGORICAL_FEATURES)]


# Build the shared numeric-scaling and categorical-encoding transformer.
def make_feature_preprocessor() -> ColumnTransformer:
    """Create the feature transformer used by all model-specific pipelines.

    The returned transformer must be fitted on the training partition only.
    Unknown categories in validation or test data are ignored.
    """
    # Run rare-category grouping before one-hot encoding.
    categorical_pipeline = Pipeline(
        steps=[
            # Collapse sparse and unseen values before creating sparse indicators.
            ("group_infrequent", AverageCategoryFrequencyGrouper()),
            (
                "one_hot",
                OneHotEncoder(
                    # New categories in later chronological partitions do not
                    # change the fitted feature layout.
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )
    # Apply numeric and categorical transformations to their respective columns.
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                MinMaxScaler(),
                list(NUMERIC_FEATURES),
            ),
            (
                "categorical",
                categorical_pipeline,
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )


# Select the model inputs and keep the target separate.
def prepare_model_features(
    data: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the target and derived target-source fields."""
    # Check that every feature required by the transformer is present.
    missing = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES) - set(data.columns)
    if missing:
        raise KeyError(f"Missing model features: {sorted(missing)}")
    if target_column not in data:
        raise KeyError(f"Missing target column: {target_column}")

    # Select only modeling inputs; targets and raw source fields stay separate.
    features = data[
        list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    ].copy()
    return features, data[target_column].copy()
