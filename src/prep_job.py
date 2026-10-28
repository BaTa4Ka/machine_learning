"""Preprocessing pipeline for chronological train, validation, and test splits."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SERVICE_TYPES = ("IC", "ICE", "RE")
REQUIRED_COLUMNS = {
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
}


def load_monthly_data(
    data_dir: Path | str,
    columns: Iterable[str],
    *,
    months: Iterable[int] = range(1, 10),
) -> pd.DataFrame:
    """Load selected columns from the monthly source files."""
    data_path = Path(data_dir)
    parts = []
    for month in months:
        file = data_path / f"data-2026-{month:02d}.parquet"
        if not file.exists():
            raise FileNotFoundError(f"Missing monthly data file: {file}")
        frame = pd.read_parquet(file, columns=list(columns))
        frame = frame[frame["train_type"].isin(SERVICE_TYPES)].copy()
        frame["source_file"] = file.name
        parts.append(frame)
    if not parts:
        raise ValueError("months must contain at least one month")
    return pd.concat(parts, ignore_index=True)


def add_target(data: pd.DataFrame) -> pd.DataFrame:
    """Create the five-class arrival-delay target."""
    result = data.copy()
    result["arrival_is_canceled"] = (
        result["arrival_is_canceled"].fillna(False).astype(bool)
    )
    result = result[
        result["arrival_is_canceled"] | result["delay_in_min"].notna()
    ].copy()
    result["target"] = np.select(
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


def add_split_time(data: pd.DataFrame) -> pd.DataFrame:
    """Create the timestamp used only for chronological splitting."""
    result = data.copy()
    result["planned_time"] = result["arrival_planned_time"].fillna(
        result["departure_planned_time"]
    )
    return result.dropna(subset=["planned_time"]).copy()


def chronological_split(
    data: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Split rows in time order into train, validation, and test partitions."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must leave rows for the test partition")
    if data.empty:
        raise ValueError("cannot split an empty dataset")

    ordered = data.sort_values("planned_time", kind="stable").reset_index(drop=True)
    train_end = int(len(ordered) * train_fraction)
    validation_end = int(
        len(ordered) * (train_fraction + validation_fraction)
    )
    partitions = {
        "train": ordered.iloc[:train_end].copy(),
        "validation": ordered.iloc[train_end:validation_end].copy(),
        "test": ordered.iloc[validation_end:].copy(),
    }
    if any(partition.empty for partition in partitions.values()):
        raise ValueError("chronological split produced an empty partition")
    return partitions


def _validate_columns(data: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise KeyError(f"Missing preprocessing columns: {sorted(missing)}")


def add_first_station(data: pd.DataFrame) -> pd.DataFrame:
    """Derive the first station using the train-line ride identifier."""
    result = data.copy()
    first_stations = (
        result[result["train_line_station_num"] == 1]
        .groupby("train_line_ride_id", dropna=False)["station_name"]
        .first()
        .rename("first_station")
    )
    result = result.drop(columns=["first_station"], errors="ignore")
    return result.join(first_stations, on="train_line_ride_id")


def preprocess_split(
    data: pd.DataFrame,
    *,
    fill_missing_line_number: Any | None = None,
    fill_missing_final_destination: bool = False,
) -> pd.DataFrame:
    """Preprocess one partition without using rows from another partition."""
    _validate_columns(data)
    result = data.copy()
    result["day_of_week"] = pd.to_datetime(result["time"]).dt.day_name()
    result = result[
        result["delay_in_min"].between(-1, 180)
        | result["delay_in_min"].isna()
    ].copy()
    result = add_first_station(result)
    result = result.dropna(subset=["first_station"]).copy()
    result = result[
        (result["train_type"] != "RE") | result["line_number"].notna()
    ].copy()

    if fill_missing_line_number is not None:
        result["line_number"] = result["line_number"].fillna(
            fill_missing_line_number
        )
    if fill_missing_final_destination:
        result["final_destination_station"] = result[
            "final_destination_station"
        ].fillna(result["station_name"])

    return result.drop(
        columns=[
            "arrival_planned_time",
            "departure_planned_time",
            "eva",
            "source_file",
            "train_line_ride_id",
        ],
        errors="ignore",
    )


def preprocess_partitions(
    partitions: dict[str, pd.DataFrame],
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Apply preprocessing independently to every named partition."""
    return {
        name: preprocess_split(partition, **kwargs)
        for name, partition in partitions.items()
    }
