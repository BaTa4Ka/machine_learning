"""Preprocessing transformations for individual data partitions."""

from collections.abc import Iterable
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {
    "arrival_planned_time",
    "departure_planned_time",
    "time",
    "delay_in_min",
    "train_type",
    "line_number",
    "station_name",
    "train_number",
    "train_line_station_num",
}


def _validate_columns(data: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise KeyError(f"Missing preprocessing columns: {sorted(missing)}")


def add_first_station(
    data: pd.DataFrame,
    *,
    output_column: str = "first_station",
    group_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Add the station where each train-line ride starts.

    IC and ICE rows are grouped by ``train_number``. RE rows are grouped by
    ``line_number``. Train type is included in both keys so values from
    different service types cannot be matched.
    """
    result = data.copy()
    if group_columns is None:
        result["_first_station_group"] = result["train_number"].astype("string")
        re_rows = result["train_type"].eq("RE")
        result.loc[re_rows, "_first_station_group"] = result.loc[
            re_rows, "line_number"
        ]
        result["_first_station_group"] = result["_first_station_group"].astype(
            "string"
        )
        group_columns = ["train_type", "_first_station_group"]
    group_columns = list(group_columns)
    missing = set(group_columns) - set(result.columns)
    if missing:
        raise KeyError(f"Missing first-station grouping columns: {sorted(missing)}")

    first_rows = result[result["train_line_station_num"] == 1]
    first_station_by_group = (
        first_rows.groupby(group_columns, dropna=False)["station_name"]
        .first()
        .rename(output_column)
    )
    result = result.drop(columns=[output_column], errors="ignore")
    result = result.join(first_station_by_group, on=group_columns)
    if "_first_station_group" in result:
        result = result.drop(columns="_first_station_group")
    return result


def add_planned_time_features(
    data: pd.DataFrame,
    *,
    add_weekday: bool = True,
) -> pd.DataFrame:
    """Add a schedule-time fallback and weekday feature.

    Arrival schedule time is preferred; departure schedule time fills only
    missing arrival values. Rows with neither value remain for later removal.
    """
    result = data.copy()
    result["planned_time"] = result["arrival_planned_time"].fillna(
        result["departure_planned_time"]
    )
    if add_weekday:
        result["day_of_week"] = pd.to_datetime(result["time"]).dt.day_name()
    return result


def preprocess_split(
    data: pd.DataFrame,
    *,
    fill_missing_line_number: Any | None = None,
    fill_missing_final_destination: bool = False,
) -> pd.DataFrame:
    """Clean and enrich one train, validation, or test partition.

    Rows without either planned arrival or departure time, delays below -1
    minute, or delays above 180 minutes are removed. Cancelled rows with a
    missing delay are retained. Rows without a derived first station and RE
    rows without a line number are removed. Optional missing value treatments
    are applied after filtering. The source arrival and departure schedule
    columns are replaced by ``planned_time``.
    """
    _validate_columns(data)
    result = add_planned_time_features(data)
    result = result.dropna(subset=["planned_time"]).copy()
    result["day_of_week"] = pd.to_datetime(result["time"]).dt.day_name()
    valid_delay = result["delay_in_min"].between(-1, 180) | result[
        "delay_in_min"
    ].isna()
    result = result[valid_delay]
    if "first_station" not in result:
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
        if "final_destination_station" not in result:
            raise KeyError(
                "final_destination_station is required when "
                "fill_missing_final_destination=True"
            )
        result["final_destination_station"] = result[
            "final_destination_station"
        ].fillna(result["station_name"])
    result = result.drop(
        columns=["arrival_planned_time", "departure_planned_time"]
    )
    return result


def preprocess_splits(
    splits: dict[str, pd.DataFrame],
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Apply :func:`preprocess_split` independently to named partitions."""
    return {
        name: preprocess_split(partition, **kwargs)
        for name, partition in splits.items()
    }
