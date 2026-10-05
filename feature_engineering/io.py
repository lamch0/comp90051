"""Load raw data and export only the requested training handoff files."""
import json
from pathlib import Path
import pandas as pd
from .schema import ID, REQUIRED
from .builder import build
from .selection import compact_experiments


def run(args):
    """Generate features, labels and experiment lists; optionally assign folds.

    Raw inputs are never modified. ID is a join key, not a predictor. No model,
    imputer, encoder, scaler or data-dependent selector is fitted.
    """
    source = Path(args.data_dir)
    frames = {name: pd.read_csv(source / f'{name}.csv', low_memory=False)
              for name in REQUIRED}
    full, _, experiments, _, _ = build(
        frames['application_train'], frames['installments_payments'],
        frames['bureau'], frames['bureau_balance'], args.days_per_month)
    compact, _ = compact_experiments(experiments)
    columns = list(dict.fromkeys(c for cols in compact.values() for c in cols))
    features = full[columns]
    labels = frames['application_train'].set_index(ID).TARGET
    if not features.index.is_unique or not features.index.equals(labels.index):
        raise ValueError('Unsafe applicant/label alignment')
    if features.columns.duplicated().any() or {ID, 'TARGET'} & set(features.columns):
        raise ValueError('Unsafe predictor columns')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    features.to_csv(out / 'features_compact.csv')
    labels.to_csv(out / 'labels.csv')
    (out / 'experiments_compact.json').write_text(json.dumps(compact, indent=2))
    if args.nested_folds:
        from .folds import nested_fold_plan
        nested_fold_plan(labels).to_csv(out / 'nested_folds.csv')
    print(f'Exported {len(features):,} applicants and {len(columns)} predictors to {out}')
