"""Utilities for evaluating fitted classifiers on a held-out test set."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


class PredictiveModel(Protocol):
    """The part of a fitted classifier required by the evaluation helpers."""

    def predict(self, X: Any) -> Any:
        """Return predictions for the supplied feature matrix."""


def macro_f1_test(
    model: PredictiveModel,
    X_test: Any,
    y_test: Sequence[Any],
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate one fitted classifier on a held-out test set.

    The classifier must already be fitted. This function does not fit or alter the
    model, which prevents test data from influencing training. Macro-F1 gives every
    target class equal weight and uses ``zero_division=0`` when a classifier never
    predicts a class.

    Parameters
    ----------
    model:
        A fitted classifier or baseline exposing ``predict``.
    X_test:
        Test features.
    y_test:
        True test labels.
    model_name:
        Optional display name for the returned result.

    Returns
    -------
    dict[str, Any]
        The model name, number of test rows, macro-F1, and accuracy.

    Raises
    ------
    ValueError
        If the test labels are empty or the number of predictions does not match
        the number of labels.
    AttributeError
        If ``model`` does not provide a ``predict`` method.
    """
    if not hasattr(model, "predict"):
        raise AttributeError("model must provide a predict(X_test) method")

    labels = list(y_test)
    if not labels:
        raise ValueError("y_test must contain at least one label")

    predictions = model.predict(X_test)
    if len(predictions) != len(labels):
        raise ValueError(
            "The number of predictions must match the number of test labels: "
            f"{len(predictions)} != {len(labels)}"
        )

    return {
        "model": model_name or type(model).__name__,
        "n_test": len(labels),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "accuracy": accuracy_score(labels, predictions),
    }


def evaluate_models(
    models: Mapping[str, PredictiveModel],
    X_test: Any,
    y_test: Sequence[Any],
) -> pd.DataFrame:
    """Compare fitted baselines and models on the same held-out test set.

    Parameters
    ----------
    models:
        Mapping from the name to a fitted classifier or baseline.
    X_test:
        Test features shared by all models.
    y_test:
        True test labels.

    Returns
    -------
    pandas.DataFrame
        One row per model, sorted by descending macro-F1 and then accuracy.
    """
    if not models:
        raise ValueError("models must contain at least one fitted model")

    results = [
        macro_f1_test(model, X_test, y_test, model_name=name)
        for name, model in models.items()
    ]
    return (
        pd.DataFrame(results)
        .sort_values(["macro_f1", "accuracy"], ascending=False)
        .reset_index(drop=True)
    )
