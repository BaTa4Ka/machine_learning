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
missing monthly files. The notebook calls the reusable downloader in
[`src/download_data.py`](src/download_data.py), checks each file before downloading it,
and can be safely run again.

## Environment setup on Windows

Use Python 3.14 or a compatible newer 3.x release. The current environment is tested
with Python 3.14.7. Check the installed interpreter before creating the environment:

```powershell
py --version
```

Open PowerShell in the repository root and create the virtual environment:

```powershell
py -3.14 -m venv .venv
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

With `.venv` activated, start Jupyter from the `notebooks` directory so the notebook can
resolve the repository root:

```powershell
Set-Location notebooks
jupyter lab
```

Open `setup.ipynb` and run the code cell. It checks for
`data-2026-01.parquet` through `data-2026-09.parquet` and downloads only files that are
missing. Incomplete downloads are removed automatically.

After the data is available, open
[`data_audit.ipynb`](notebooks/data_audit.ipynb) to review the dataset shape, variable
types, missing values, duplicates, target distribution, and potential delay outliers.

## Preprocessing and evaluation design

The preprocessing workflow is implemented in
[`src/prep_job.py`](src/prep_job.py) and demonstrated in
[`notebooks/preprocessing.ipynb`](notebooks/preprocessing.ipynb).

The data is ordered by a unified `planned_time` value. `arrival_planned_time` is used
when available, with `departure_planned_time` as a fallback. Rows with neither
timestamp are removed because they cannot be placed reliably in chronological order.

The remaining data is split chronologically into:

- 70% training data;
- 15% validation data;
- 15% test data.

Preprocessing is applied independently after the split. It removes delays below
`-1` minute and above `180` minutes, derives the first station, removes rows without
a derivable first station, removes RE rows without a line number, and creates a
`day_of_week` feature from `time`. The original arrival and departure schedule
columns are replaced by `planned_time`.

The training partition is used to fit models, the validation partition is reserved
for model selection and tuning, and the test partition is kept untouched until final
evaluation. Random cross-validation should not be used across the full dataset because
it can mix earlier and later observations. If cross-validation is needed, it should
preserve chronological order within the training period.

The baseline workflow is in
[`notebooks/baseline.ipynb`](notebooks/baseline.ipynb). It uses the same preprocessing
pipeline and reports accuracy and macro-F1 separately for IC, ICE, and RE. Macro-F1
is the primary metric because it gives each delay class equal weight, while accuracy
can be dominated by the most common class.

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
