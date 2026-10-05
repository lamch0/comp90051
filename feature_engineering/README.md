# Home Credit feature engineering

This package generates only:
- features_compact.csv: one applicant per row, recommended predictor union and ID.
- labels.csv: ID and TARGET, separate from predictors.
- experiments_compact.json: exact predictor names for each experiment.
- nested_folds.csv: optional shared 10-outer/3-inner stratified assignments.

It contains no dictionary/report generation, model training or fitted population
preprocessing. Raw data and previously generated output are not included.

## Run

Run from the project root (the directory containing feature_engineering/).
Place application_train.csv, installments_payments.csv, bureau.csv and
bureau_balance.csv in raw_data/ at that project root.

```bash
python3 -m pip install -r feature_engineering/requirements.txt
python3 -m feature_engineering.pipeline --data-dir raw_data --output-dir outputs
```

To also generate nested_folds.csv:

```bash
python3 -m feature_engineering.pipeline --data-dir raw_data --output-dir outputs --nested-folds
```

The optional assignment module is folds.py; it is imported only with --nested-folds.
All applicants are retained. Use a fresh output directory to avoid confusing files
from earlier executions. The pipeline never deletes existing output files.

## Included code

pipeline.py is the entry point; io.py loads data and exports the selected artifacts.
schema.py defines columns and windows; application_columns.json contains only the
70-field allowlist. cleaning.py cleans and validates records. features.py constructs
source-level measurements. temporal.py computes window aggregates and dynamics.
builder.py assembles one row per applicant. selection.py applies fixed semantic
simplification. experiments.py defines experiment column lists. __init__.py defines
the package. folds.py is the optional split generator. No documentation catalog is
required to run this version.

## Interpretation and training boundary

On the supplied full dataset, the union contains 329 predictors: 70 highlighted
raw application fields and 41 engineered feature types repeated across windows.
Choose one experiment's columns; do not use the whole union automatically.
ID is only a join key; TARGET is only the label. No history is not evidence of
on-time payment. Bureau snapshots describe opening cohorts, not debt trajectories.
Imputation, encoding, scaling and empirical selection must be fitted later inside
training folds. This submission stops before those operations and model training.
