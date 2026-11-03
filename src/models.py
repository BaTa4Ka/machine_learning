"""Model factories and validation-based model selection."""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    from .model_evaluation import macro_f1_test
except ImportError:
    from model_evaluation import macro_f1_test


def make_multinomial_logistic_regression(
    *,
    C: float = 1.0,
    max_iter: int = 100,
    random_state: int = 42,
) -> LogisticRegression:
    """Create a class-balanced logistic-regression classifier for sparse features."""
    if C <= 0:
        raise ValueError("C must be greater than zero")
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than zero")

    return LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state,
        solver="saga",
    )


def select_by_validation_macro_f1(
    models: Mapping[str, Any],
    X_train: Any,
    y_train: Sequence[Any],
    X_validation: Any,
    y_validation: Sequence[Any],
) -> tuple[str, Any, pd.DataFrame]:
    """Fit candidates on training data and select the best validation macro-F1."""
    if not models:
        raise ValueError("models must contain at least one candidate")

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(
            macro_f1_test(
                model,
                X_validation,
                y_validation,
                model_name=name,
            )
        )

    validation_results = (
        pd.DataFrame(results)
        .sort_values(["macro_f1", "accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    selected_name = validation_results.iloc[0]["model"]
    return selected_name, models[selected_name], validation_results
