"""Resumable, from-scratch nested CV. Run as python -m training.run."""
import argparse
import hashlib
import importlib.metadata
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import binary_metrics
from .models import make_model
from .preprocessing import FoldPreprocessor
from .selection import select_numeric_training_columns


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    """Atomic replacement prevents interrupted writes looking like checkpoints."""
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def save_csv(path, frame):
    temporary = path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_inputs(data_dir):
    """Validate cohort identity and every split; never construct new splits here."""
    root = Path(data_dir)
    X = pd.read_csv(root/'features_compact.csv', index_col='SK_ID_CURR', low_memory=False)
    labels = pd.read_csv(root/'labels.csv', index_col='SK_ID_CURR')
    folds = pd.read_csv(root/'nested_folds.csv', index_col='SK_ID_CURR')
    if not all(f.index.is_unique for f in [X, labels, folds]) or X.columns.duplicated().any():
        raise ValueError('Duplicate IDs or predictor columns')
    if not set(X.index) == set(labels.index) == set(folds.index):
        raise ValueError('Features, labels and folds must have identical applicant IDs')
    labels, folds = labels.loc[X.index], folds.loc[X.index]
    if 'TARGET' in X or not labels.TARGET.isin([0,1]).all():
        raise ValueError('TARGET must be binary and excluded from predictors')
    if not set(folds.outer_fold.unique()) == set(range(10)):
        raise ValueError('Exactly ten outer folds numbered 0..9 required')
    y = labels.TARGET.astype(int)
    for k in range(10):
        test = folds.outer_fold == k
        inner = folds[f'inner_for_outer_{k}']
        if not (inner[test] == -1).all() or set(inner[~test].unique()) != {0,1,2}:
            raise ValueError(f'Invalid nested assignments for outer fold {k}')
        for mask in [test, ~test, *[(~test)&(inner == j) for j in range(3)],
                     *[(~test)&(inner != j) for j in range(3)]]:
            if set(y[mask].unique()) != {0,1}:
                raise ValueError('Every training/evaluation subset must contain both classes')
    experiments = json.loads((root/'experiments_compact.json').read_text())
    for name, cols in experiments.items():
        if len(cols) != len(set(cols)) or not set(cols) <= set(X):
            raise ValueError(f'Invalid experiment: {name}')
    return X, y, folds, experiments


def prepare(train, valid, screen):
    """Learn selection, imputation, encoding and scaling only from train."""
    keep, audit = select_numeric_training_columns(train) if screen else (list(train), {})
    prep = FoldPreprocessor().fit(train[keep])
    a = prep.transform(train[keep]).to_numpy(dtype=np.float32)
    b = prep.transform(valid[keep]).to_numpy(dtype=np.float32)
    if not a.shape[1]:
        raise ValueError('No nonconstant predictors remain')
    return a, b, {'selected_raw_count': len(keep), 'encoded_count': a.shape[1],
                  'omitted': audit}


def fit_predict(name, params, a, b, labels, seed, threads):
    """Limit CPU concurrency and record convergence warnings rather than hide them."""
    from threadpoolctl import threadpool_limits
    model = make_model(name, params, seed, threads)
    start = time.perf_counter()
    with threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(a, labels)
        probabilities = model.predict_proba(b)[:, 1]
    return probabilities, time.perf_counter()-start, [str(w.message) for w in caught]


def summarise(out):
    """Fold-level means and sample SD provide descriptive error bars, not CIs."""
    rows = [json.loads(p.read_text()) for p in sorted(out.glob('*/*/outer_*/result.json'))]
    if not rows:
        return
    flat = [{**{k:r[k] for k in ['experiment','model','outer_fold','seconds']},
             **r['metrics']} for r in rows]
    frame = pd.DataFrame(flat)
    save_csv(out/'outer_results.csv', frame)
    summary = []
    metrics = ['roc_auc','average_precision','log_loss','brier_score',
               'accuracy','precision','recall','f1']
    for (experiment, model), group in frame.groupby(['experiment','model']):
        selected = [r['best_params'] for r in rows if r['experiment']==experiment and r['model']==model]
        counts = {}
        for params in selected:
            key = json.dumps(params, sort_keys=True)
            counts[key] = counts.get(key, 0)+1
        save_json(out/experiment/model/'parameter_selection_counts.json', counts)
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            summary.append({'experiment':experiment, 'model':model, 'metric':metric,
                            'completed_outer_folds':len(values), 'complete':len(values)==10,
                            'mean':float(np.mean(values)),
                            'sd':float(np.std(values,ddof=1)) if len(values)>1 else None})
    save_csv(out/'summary.csv', pd.DataFrame(summary))


def run(args):
    X, y, folds, experiments = load_inputs(args.data_dir)
    grid_path = Path(args.grids)
    grids = json.loads(grid_path.read_text())
    for name in args.models:
        if name not in grids or len(grids[name])<3:
            raise ValueError('Each model needs at least three distinct candidate settings')
        if len({json.dumps(p,sort_keys=True) for p in grids[name]}) != len(grids[name]):
            raise ValueError('Duplicate grid candidates')
    if not set(args.experiments) <= set(experiments):
        raise ValueError('Unknown experiment name')
    root, out = Path(args.data_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Selected folds/models/experiments may expand during resume, but data,
    # implementation, grids and numerical environment must match the prior run.
    versions = {}
    for package in ['numpy','pandas','scikit-learn','lightgbm','threadpoolctl']:
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    identity = {'inputs':{n:fingerprint(root/n) for n in
                ['features_compact.csv','labels.csv','experiments_compact.json','nested_folds.csv']},
                'code':{p.name:fingerprint(p) for p in Path(__file__).parent.glob('*.py')},
                'grids':grids, 'seed':args.seed, 'threads':args.threads,
                'screen':not args.no_correlation_screen, 'versions':versions,
                'selection_metric':'roc_auc', 'classification_threshold':.5}
    manifest_path = out/'run_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != identity:
            raise ValueError('Resume identity changed. Use a new output directory.')
    else:
        save_json(manifest_path, identity)
    for experiment in args.experiments:
        predictors = X[experiments[experiment]]
        for name in args.models:
            candidates = grids[name]
            for k in args.outer_folds:
                directory = out/experiment/name/f'outer_{k}'
                directory.mkdir(parents=True, exist_ok=True)
                if (directory/'result.json').exists():
                    print(f'Skip completed {experiment}/{name}/outer_{k}', flush=True)
                    continue
                print(f'Start {experiment}/{name}/outer_{k}', flush=True)
                started = time.perf_counter()
                outer_train = folds.outer_fold != k
                inner = folds[f'inner_for_outer_{k}']
                for j in range(3):
                    pending = [c for c in range(len(candidates))
                               if not (directory/f'candidate_{c}_inner_{j}.json').exists()]
                    if not pending:
                        continue
                    train = outer_train & (inner != j)
                    valid = outer_train & (inner == j)
                    a, b, prep = prepare(predictors.loc[train], predictors.loc[valid],
                                         not args.no_correlation_screen)
                    for c in pending:
                        p, seconds, messages = fit_predict(name, candidates[c], a, b, y[train],
                                                          args.seed+k*10+j, args.threads)
                        record = {'params':candidates[c], 'inner_fold':j,
                                  'metrics':binary_metrics(y[valid],p), 'seconds':seconds,
                                  'warnings':messages, 'preprocessing':prep}
                        save_json(directory/f'candidate_{c}_inner_{j}.json', record)
                        print(f'  inner={j} candidate={c} AUC={record["metrics"]["roc_auc"]:.4f} '
                              f'fit+predict={seconds:.1f}s', flush=True)
                    del a,b
                scores = []
                for c in range(len(candidates)):
                    scores.append(float(np.mean([json.loads(
                        (directory/f'candidate_{c}_inner_{j}.json').read_text())['metrics']['roc_auc']
                        for j in range(3)])))
                best = int(np.argmax(scores))  # Fixed input order breaks exact ties.
                test = ~outer_train
                a, b, prep = prepare(predictors.loc[outer_train], predictors.loc[test],
                                     not args.no_correlation_screen)
                p, fit_seconds, messages = fit_predict(name, candidates[best], a, b,
                                                       y[outer_train], args.seed+k,
                                                       args.threads)
                save_csv(directory/'predictions.csv', pd.DataFrame({
                    'SK_ID_CURR':X.index[test], 'TARGET':y[test].to_numpy(),
                    'probability':p, 'outer_fold':k}))
                record = {'experiment':experiment, 'model':name, 'outer_fold':k,
                          'train_rows':int(outer_train.sum()), 'test_rows':int(test.sum()),
                          'best_params':candidates[best], 'candidate_mean_auc':scores,
                          'boundary_selected':best in [0,len(candidates)-1],
                          'metrics':binary_metrics(y[test],p), 'preprocessing':prep,
                          'warnings':messages, 'outer_fit_predict_seconds':fit_seconds,
                          'seconds':time.perf_counter()-started}
                save_json(directory/'result.json', record)
                summarise(out)
                print(f'Finished {experiment}/{name}/outer_{k}: '
                      f'AUC={record["metrics"]["roc_auc"]:.4f}, '
                      f'elapsed this invocation={record["seconds"]:.1f}s', flush=True)
                del a,b
    summarise(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default='training_results')
    parser.add_argument('--experiments', nargs='+', default=['E9_dynamics24'])
    parser.add_argument('--models', nargs='+', choices=['lr','cart','lightgbm'], default=['lr','cart','lightgbm'])
    parser.add_argument('--outer-folds', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--grids', default=str(Path(__file__).with_name('grids.json')))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-correlation-screen', action='store_true')
    args = parser.parse_args()
    if args.threads<1 or len(set(args.outer_folds))!=len(args.outer_folds) or not set(args.outer_folds)<=set(range(10)):
        parser.error('Positive threads and unique outer folds in 0..9 required')
    run(args)


if __name__ == '__main__':
    main()
