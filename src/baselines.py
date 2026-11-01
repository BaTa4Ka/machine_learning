"""Simple baseline classifiers."""

from typing import Any

import numpy as np


class RandomClassShareClassifier:
    """Predict labels randomly according to their fitted class frequencies."""

    # Store the seed used to reproduce random baseline predictions.
    def __init__(self, *, random_state: int | None = 42) -> None:
        self.random_state = random_state

    # Learn class probabilities from the labels in the fitting partition.
    def fit(self, _X: Any, y: Any) -> "RandomClassShareClassifier":
        labels = np.asarray(y)
        if labels.size == 0:
            raise ValueError("y must contain at least one label")

        # Count each class and convert counts into sampling probabilities.
        self.classes_, counts = np.unique(labels, return_counts=True)
        self.class_probabilities_ = counts / counts.sum()
        self._rng = np.random.default_rng(self.random_state)
        return self

    # Draw one class label for every requested prediction row.
    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "classes_"):
            raise RuntimeError("RandomClassShareClassifier must be fitted first")
        return self._rng.choice(
            self.classes_,
            size=len(X),
            p=self.class_probabilities_,
        )


class MajorityClassClassifier:
    """Always predict the most frequent fitted class."""

    # Learn the class that appears most often in the fitting partition.
    def fit(self, _X: Any, y: Any) -> "MajorityClassClassifier":
        labels = np.asarray(y)
        if labels.size == 0:
            raise ValueError("y must contain at least one label")

        # Keep class counts for inspection and select the most common class.
        classes, counts = np.unique(labels, return_counts=True)
        self.classes_ = classes
        self.class_counts_ = counts
        self.majority_class_ = classes[counts.argmax()]
        return self

    # Repeat the fitted majority class for every requested prediction row.
    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "majority_class_"):
            raise RuntimeError("MajorityClassClassifier must be fitted first")
        return np.full(len(X), self.majority_class_)
