"""Download the monthly source files used by the project."""

from pathlib import Path
from urllib.request import urlretrieve

DATASET_URL = (
    "https://huggingface.co/datasets/piebro/deutsche-bahn-data/"
    "resolve/main/monthly_processed_data"
)


def download_monthly_data(
    data_dir: Path | str,
    *,
    start_month: int = 1,
    end_month: int = 9,
) -> list[Path]:
    """Ensure the requested monthly Parquet files exist locally.

    Existing files are left unchanged. A failed download is removed so a
    partially downloaded Parquet file cannot be mistaken for valid data.
    """
    if not 1 <= start_month <= end_month <= 12:
        raise ValueError("months must satisfy 1 <= start_month <= end_month <= 12")

    destination_dir = Path(data_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = []

    for month in range(start_month, end_month + 1):
        filename = f"data-2026-{month:02d}.parquet"
        destination = destination_dir / filename

        if destination.exists():
            print(f"Already exists: {destination}")
            downloaded_files.append(destination)
            continue

        url = f"{DATASET_URL}/{filename}?download=true"
        temporary_destination = destination.with_suffix(".parquet.download")
        print(f"Downloading {filename}...")
        try:
            urlretrieve(url, temporary_destination)
            temporary_destination.replace(destination)
        except Exception:
            temporary_destination.unlink(missing_ok=True)
            raise

        print(f"Saved to {destination}")
        downloaded_files.append(destination)

    return downloaded_files
