"""Categorical preprocessing for boosted tree models."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from .sklearn_preprocessing import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        TARGET_COLUMN,
    )
except ImportError:
    from sklearn_preprocessing import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        TARGET_COLUMN,
    )


class BoostedTreeFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Keep numeric features and aligned pandas categorical features.

    Category frequencies and category vocabularies are learned from training
    data only. Rare, missing, and later unseen values are mapped explicitly so
    every partition has compatible pandas categorical dtypes.
    """

    def __init__(
        self,
        *,
        rare_value: str = "__rare__",
        missing_value: str = "__missing__",
    ) -> None:
        self.rare_value = rare_value
        self.missing_value = missing_value

    def fit(
        self,
        X: pd.DataFrame,
        y: object = None,
    ) -> "BoostedTreeFeaturePreprocessor":
        frame = self._validate_input(X)
        self.common_categories_: dict[str, set[str]] = {}
        self.categories_: dict[str, list[str]] = {}

        for column in CATEGORICAL_FEATURES:
            values = self._normalise_categories(frame[column])
            counts = values.value_counts()
            average_count = len(frame) / len(counts)
            common = set(counts[counts >= average_count].index)
            self.common_categories_[column] = common
            category_values = sorted(common | {self.rare_value, self.missing_value})
            self.categories_[column] = category_values
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "categories_"):
            raise RuntimeError("BoostedTreeFeaturePreprocessor must be fitted first")

        frame = self._validate_input(X).copy()
        for column in CATEGORICAL_FEATURES:
            values = self._normalise_categories(frame[column])
            values = values.where(
                values.isin(self.common_categories_[column])
                | values.eq(self.missing_value),
                self.rare_value,
            )
            frame[column] = pd.Categorical(
                values,
                categories=self.categories_[column],
            )
        return frame[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Boosted-tree features must be provided as a DataFrame")
        required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        missing = required - set(X.columns)
        if missing:
            raise KeyError(f"Missing model features: {sorted(missing)}")
        return X[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]

    def _normalise_categories(self, values: pd.Series) -> pd.Series:
        return values.astype("string").fillna(self.missing_value)


def prepare_boosted_tree_features(
    data: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate boosted-tree-compatible features from the target."""
    if target_column not in data:
        raise KeyError(f"Missing target column: {target_column}")
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Missing model features: {sorted(missing)}")
    return (
        data[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)].copy(),
        data[target_column].copy(),
    )


def make_boosted_tree_preprocessor() -> BoostedTreeFeaturePreprocessor:
    """Create the non-one-hot preprocessor for boosted tree models."""
    return BoostedTreeFeaturePreprocessor()
