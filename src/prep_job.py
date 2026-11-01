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


# Load the selected columns from each monthly Parquet file.
def load_monthly_data(
    data_dir: Path | str,
    columns: Iterable[str],
    *,
    months: Iterable[int] = range(1, 10),
) -> pd.DataFrame:
    """Load selected columns from the monthly source files."""
    # Resolve the directory once so every monthly file uses the same location.
    data_path = Path(data_dir)
    # Collect filtered monthly frames before concatenating them.
    parts = []
    for month in months:
        # Build the expected filename from the requested month number.
        file = data_path / f"data-2026-{month:02d}.parquet"
        if not file.exists():
            raise FileNotFoundError(f"Missing monthly data file: {file}")
        # Read only the columns required by the preprocessing workflow.
        frame = pd.read_parquet(file, columns=list(columns))
        # Keep only the supported long-distance and regional train types.
        frame = frame[frame["train_type"].isin(SERVICE_TYPES)].copy()
        frame["source_file"] = file.name
        parts.append(frame)
    if not parts:
        raise ValueError("months must contain at least one month")
    return pd.concat(parts, ignore_index=True)


# Convert cancellation and delay information into the modelling target.
def add_target(data: pd.DataFrame) -> pd.DataFrame:
    """Create the five-class arrival-delay target."""
    # Work on a copy so the raw input frame is not modified.
    result = data.copy()
    # Treat missing cancellation flags as ordinary non-cancelled observations.
    result["arrival_is_canceled"] = (
        result["arrival_is_canceled"].fillna(False).astype(bool)
    )
    result = result[
        result["arrival_is_canceled"] | result["delay_in_min"].notna()
    ].copy()
    # Apply the ordered thresholds so the first matching class wins.
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


# Create the timestamp used to order rows before splitting.
def add_split_time(data: pd.DataFrame) -> pd.DataFrame:
    """Create the timestamp used only for chronological splitting."""
    # Prefer arrival time and use departure time only as a fallback.
    result = data.copy()
    result["planned_time"] = result["arrival_planned_time"].fillna(
        result["departure_planned_time"]
    )
    return result.dropna(subset=["planned_time"]).copy()


# Divide the timestamp-sorted data into chronological partitions.
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

    # Stable sorting preserves the original order when timestamps are equal.
    ordered = data.sort_values("planned_time", kind="stable").reset_index(drop=True)
    # Calculate row boundaries from the requested partition proportions.
    train_end = int(len(ordered) * train_fraction)
    validation_end = int(
        len(ordered) * (train_fraction + validation_fraction)
    )
    # Copy each partition so later preprocessing remains independent.
    partitions = {
        "train": ordered.iloc[:train_end].copy(),
        "validation": ordered.iloc[train_end:validation_end].copy(),
        "test": ordered.iloc[validation_end:].copy(),
    }
    if any(partition.empty for partition in partitions.values()):
        raise ValueError("chronological split produced an empty partition")
    return partitions


# Check that the raw columns required by preprocessing are available.
def _validate_columns(data: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise KeyError(f"Missing preprocessing columns: {sorted(missing)}")


# Add the first station associated with each train-line ride.
def add_first_station(data: pd.DataFrame) -> pd.DataFrame:
    """Derive the first station using the train-line ride identifier."""
    # Identify station-one rows because they represent each ride's origin.
    result = data.copy()
    first_stations = (
        result[result["train_line_station_num"] == 1]
        .groupby("train_line_ride_id", dropna=False)["station_name"]
        .first()
        .rename("first_station")
    )
    # Remove a stale derived column before joining the newly calculated values.
    result = result.drop(columns=["first_station"], errors="ignore")
    return result.join(first_stations, on="train_line_ride_id")


# Apply domain-specific cleaning and feature creation to one partition.
def preprocess_split(
    data: pd.DataFrame,
    *,
    fill_missing_line_number: Any | None = None,
    fill_missing_final_destination: bool = False,
) -> pd.DataFrame:
    """Preprocess one partition without using rows from another partition."""
    _validate_columns(data)
    # Keep partition processing isolated from all other partitions.
    result = data.copy()
    # Derive calendar features from the observation and planned timestamps.
    result["day_of_week"] = pd.to_datetime(result["time"]).dt.day_name()
    result["planned_hour"] = pd.to_datetime(result["planned_time"]).dt.hour
    # Exclude extreme delay values while retaining canceled rows with missing delay.
    result = result[
        result["delay_in_min"].between(-1, 180)
        | result["delay_in_min"].isna()
    ].copy()
    # Add route-origin information before removing incomplete journeys.
    result = add_first_station(result)
    result = result.dropna(subset=["first_station"]).copy()
    result = result[
        (result["train_type"] != "RE") | result["line_number"].notna()
    ].copy()

    # Apply optional replacements only when explicitly requested by the caller.
    if fill_missing_line_number is not None:
        result["line_number"] = result["line_number"].fillna(
            fill_missing_line_number
        )
    if fill_missing_final_destination:
        result["final_destination_station"] = result[
            "final_destination_station"
        ].fillna(result["station_name"])

    # Remove timestamps and technical identifiers not used by the model.
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


# Run the same independent preprocessing steps for every partition.
def preprocess_partitions(
    partitions: dict[str, pd.DataFrame],
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Apply preprocessing independently to every named partition."""
    # Pass the caller's optional preprocessing settings to each partition.
    return {
        name: preprocess_split(partition, **kwargs)
        for name, partition in partitions.items()
    }
