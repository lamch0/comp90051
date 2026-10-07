# Training LR, depth-constrained CART and LightGBM

Run from the repository root. The four inputs must share exactly the same applicant
IDs. No raw historical tables are needed. Class labels are not used to construct
features. This module is separate from feature_engineering/ and does not modify it.

## Install and benchmark

```bash
python -m pip install -r training/requirements.txt
python -m training.run --data-dir outputs/pretraining --output-dir training_results \
  --experiments E9_dynamics24 --models lr cart lightgbm --outer-folds 0 --threads 2
```

This is one outer fold, with three inner folds and three candidates per model:
30 fits across three models. It is a runtime/debugging benchmark, not the final
assessment. For a shorter initial check, specify `--models cart`.

## Complete the planned comparisons

```bash
python -m training.run --data-dir outputs/pretraining --output-dir training_results \
  --experiments E0 E1 E4 E7_24m E8_multiscale24 E9_dynamics24 \
  --models lr cart lightgbm --threads 2
```

This schedules 1,800 fits. The benchmark's completed fits are skipped. You can
instead run models or experiments in separate sequential invocations using the
same output directory. Do not have multiple processes write the same directory.
To compare dynamic mechanisms, supply experiments_mechanism.json as a renamed
experiments_compact.json in a separate handoff directory with the same other three
files, and use a NEW results directory. Data files are hashed, so changing a column
list or input file cannot silently reuse results from another configuration.

## How the implementation works

- Uses the provided 10-outer/3-inner fold assignments. No third-party CV or search.
- Fits correlation screening, median imputation, missing indicators, categorical
  vocabulary, scaling and encoded-constant removal separately on each inner train.
- Caches only inner fit scores, not data transformations across different splits.
- Selects candidates by mean inner ROC-AUC; exact ties favour the first candidate.
- Refits preprocessing and a fresh model on outer train, then predicts outer test.
- Evaluates ROC-AUC, average precision, log loss, Brier score, accuracy, precision,
  recall, F1 and confusion counts using custom NumPy calculations.
- The classification threshold is prespecified at 0.5. On imbalanced data, recall
  may be low; ranking/probability metrics are the primary results. Average precision
  is a step-weighted PR summary, not trapezoidal PR area.
- Uses ordinary class weighting (none) consistently; no oversampling or synthetic data.
- Runs on CPU, with a default two-thread cap. Dense one-hot matrices may need several
  GB of RAM. There is no claim of fitting the entire suite into a free Colab session.
- LightGBM explicitly enables GOSS and EFB, with 300 trees and learning rate 0.05.
  CART uses Gini impurity and at least 50 samples per leaf. LR uses L2 and LBFGS.
  These fixed settings are part of the method, not tuned on outer test.

## Hyperparameter grids

Edit grids.json BEFORE running. LR tunes C (inverse regularisation), CART tunes
max_depth, and LightGBM tunes num_leaves. Three initial values are supplied each.
They are starting hypotheses, not established optima. Check inner score profiles
and parameter-selection frequencies. The specification asks for mostly interior
choices; revise a boundary-heavy range using inner results only, not outer metrics.
Use a NEW output directory after changing grids. Never select among grids using
outer scores; if outer results have already been inspected, report that limitation
rather than presenting another run as untouched held-out evidence.

## Saved outputs and resuming

- run_manifest.json: input/code hashes, software versions, grids and fixed settings.
- <experiment>/<model>/outer_<k>/candidate_<c>_inner_<j>.json: inner scores,
  time, preprocessing removals and warnings. Written after each completed fit.
- .../predictions.csv: outer-test IDs, labels and probabilities.
- .../result.json: selected settings, test metrics and preprocessing details;
  this is the outer-fold completion marker.
- outer_results.csv: one row per completed model/experiment/outer fold.
- summary.csv: metric means and sample SD across completed outer folds.
- <experiment>/<model>/parameter_selection_counts.json: selected-setting counts.

The `complete` column is true only for ten completed folds. SD is descriptive
fold variability and can be plotted as an error bar; it is not a confidence interval.
Timing fields describe time in the current invocation, so resumed jobs can understate
end-to-end wall-clock work. Model warnings, including LR convergence, are preserved
and must be reviewed. The code does not save fitted models because nested CV is
an evaluation procedure, not a final deployment fit.

Writes are atomic per file. If interrupted mid-fit, only that unfinished fit needs
repeating. Changes to inputs, code, grids, versions or core settings reject resume.
No automatic subset selection: keep the shared cohort and folds consistent.

## Colab

Open colab_training.ipynb, enter your GitHub repository URL and Drive paths, then
execute its cells. Inputs are copied to Colab's local disk for reading; outputs go
straight to Drive. Keep the output folder for resuming. Install/clone on each new
runtime. Version changes intentionally require a fresh results directory.

## Verification

From the repository root: `python -m unittest training.test_training -v`.
Metric tests use analytical expectations (ties, perfect/reversed predictions).
The synthetic end-to-end test fits LR and CART on a tiny fixture, checks held-out
IDs, runs resume, and ensures manifest changes are rejected. LightGBM can be tested
with `python -m training.test_training --lightgbm-smoke` after installing dependencies.
No real-cohort training is performed by these tests.
