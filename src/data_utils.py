"""Shared data loading, target construction, and split utilities."""

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

SERVICE_TYPES = ("IC", "ICE", "RE")
TARGET_LABELS = (
    "on_time",
    "slight_delay",
    "late",
    "delayed",
    "severe_or_cancelled",
)


def monthly_files(data_dir: Path | str) -> list[Path]:
    """Return the January-September 2026 source files in month order."""
    files = sorted(Path(data_dir).glob("data-2026-0[1-9].parquet"))
    if len(files) != 9:
        raise FileNotFoundError(
            f"Expected 9 monthly files in {Path(data_dir).resolve()}, found {len(files)}. "
            "Run notebooks/setup.ipynb first."
        )
    return files


def load_monthly_data(
    data_dir: Path | str,
    columns: Sequence[str],
    *,
    service_types: Iterable[str] = SERVICE_TYPES,
    require_planned_time: bool = True,
) -> pd.DataFrame:
    """Load selected columns from the monthly files and filter service types."""
    selected_types = set(service_types)
    if not selected_types:
        raise ValueError("service_types must contain at least one train type")

    parts = []
    for file in monthly_files(data_dir):
        frame = pd.read_parquet(file, columns=list(columns))
        frame = frame[frame["train_type"].isin(selected_types)].copy()
        frame["source_file"] = file.name
        parts.append(frame)

    data = pd.concat(parts, ignore_index=True)
    if require_planned_time:
        data = data.dropna(subset=["arrival_planned_time"]).copy()
    return data


def add_delay_target(
    data: pd.DataFrame,
    *,
    target_column: str = "target",
) -> pd.DataFrame:
    """Return labelled rows with the five-class delay target."""
    result = data.copy()
    result["arrival_is_canceled"] = (
        result["arrival_is_canceled"].fillna(False).astype(bool)
    )
    result = result[
        result["arrival_is_canceled"] | result["delay_in_min"].notna()
    ].copy()
    result[target_column] = np.select(
        [
            result["arrival_is_canceled"],
            result["delay_in_min"] <= 1,
            result["delay_in_min"] <= 3,
            result["delay_in_min"] <= 10,
            result["delay_in_min"] <= 30,
        ],
        [
            "severe_or_cancelled",
            "on_time",
            "slight_delay",
            "late",
            "delayed",
        ],
        default="severe_or_cancelled",
    )
    return result


def chronological_split(
    data: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows chronologically into train, validation, and test sets.

    The default proportions are 70%, 15%, and 15%. Rows are ordered by their
    planned arrival time before splitting, so later observations cannot enter
    the training or validation sets.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave test rows")
    if data.empty:
        raise ValueError("Cannot split an empty dataset")
    if "arrival_planned_time" not in data:
        raise KeyError("data must contain arrival_planned_time")

    ordered = data.sort_values("arrival_planned_time", kind="stable").reset_index(
        drop=True
    )
    train_end = int(len(ordered) * train_fraction)
    validation_end = int(len(ordered) * (train_fraction + validation_fraction))

    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Chronological split produced an empty dataset partition")
    return train, validation, test
