# Predicting Deutsche Bahn arrival-delay categories

This project investigates whether the arrival-delay category of German **ICE**, **IC**, and
**RE** trains can be predicted from information available before a train reaches a station.

The project question is:

> Can schedule and service information be used to classify a train's arrival delay into
> five ordered categories, and how does performance differ between ICE, IC, and RE services?

## Dataset

The data comes from the public
[Deutsche Bahn dataset on Hugging Face](https://huggingface.co/datasets/piebro/deutsche-bahn-data).
It is derived from Deutsche Bahn timetable and delay records and is available under the
[CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/).

The project uses the monthly files for January to September 2026:

```text
data/data-2026-01.parquet
...
data/data-2026-09.parquet
```

The files are large and are intentionally excluded from version control. Run
[`notebooks/setup.ipynb`](notebooks/setup.ipynb) to create `data/` and download any
missing monthly files. The setup cell checks each file before downloading it, so it can
be safely run again.

## Environment setup on Windows

Open PowerShell in the repository root and create the virtual environment:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current user, run this once and then activate
again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Upgrade `pip` and install the pinned project dependencies from
[`requirements.txt`](requirements.txt):

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The packages are used for:

- `pandas` and `pyarrow`: reading and writing Parquet data;
- `numpy`: numerical operations and target construction;
- `scikit-learn`: baselines, models, and evaluation metrics;
- `jupyterlab`: running the notebooks.

To leave the environment:

```powershell
deactivate
```

## Running the data setup notebook

With `.venv` activated, start Jupyter from the `notebooks` directory so the setup cell's
`../data` path points to the repository's `data/` directory:

```powershell
Set-Location notebooks
jupyter lab
```

Open `setup.ipynb` and run the code cell. It checks for
`data-2026-01.parquet` through `data-2026-09.parquet` and downloads only files that are
missing.

## Responsible use and limitations

The dataset does not contain personal information and is openly licensed, so direct
privacy risk is low. However, performance may differ substantially by train type and
station because RE services dominate the data while IC services are comparatively rare.
These subgroup results should be reported rather than relying only on an overall score.

The model is intended for analysis, not operational decisions. An incorrect
"on-time" prediction could cause a traveller to plan an unsafe or overly tight
connection. The results show associations rather than causes, and patterns observed from
January-September 2026 may not generalise to other periods or disruptions.

## Licence and attribution

The source data is provided by the dataset authors and originates from Deutsche Bahn.
See the [dataset page](https://huggingface.co/datasets/piebro/deutsche-bahn-data) and
[CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/) for source terms and
attribution requirements.
