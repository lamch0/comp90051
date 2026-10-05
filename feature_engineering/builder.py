"""In-memory orchestration of cleaning, features and experiment definitions."""
import numpy as np
import pandas as pd
from .schema import ID, REQUIRED, WINDOWS
from .cleaning import require
from .features import application_features, installments, bureau_features, balance_features
from .temporal import in_window, aggregate, dynamics
from .experiments import experiment_columns

def build(app, ins, buro, bb, days_per_month=30.0):
    """Coordinate all feature construction without reading/writing files.

    Inputs are the four raw pandas DataFrames; days_per_month defaults to 30.
    Validate keys/labels, restrict history to application IDs, prepare source rows,
    aggregate whole/cumulative/disjoint windows, add dynamics.
    TARGET is validated but is not used to calculate any predictor.

    Returns a five-item tuple:
      full: applicant-indexed feature matrix (includes every representation).
      coverage: empty indexed compatibility placeholder; no diagnostics exported.
      experiments: E0--E6/E_bins mappings to predictor column names.
      groups: source/representation mappings to predictor column names.
      audit: counts of excluded or masked historical records.
    Neither cohort selection nor fitted imputation/encoding/scaling occurs here."""
    if days_per_month <= 0: raise ValueError('days_per_month must be positive')
    for name, df in [('application_train',app),('installments_payments',ins),('bureau',buro),('bureau_balance',bb)]:
        require(df, REQUIRED[name], name)
        if name != 'application_train' and 'TARGET' in df: raise ValueError(f'TARGET in history {name}')
    if app[ID].duplicated().any(): raise ValueError('Duplicate application ID')
    if not app.TARGET.isin([0,1]).all(): raise ValueError('TARGET must be binary and non-null')
    ids = pd.Index(app[ID], name=ID)
    audit = {}
    ins = ins.loc[ins[ID].isin(ids)].copy(); buro = buro.loc[buro[ID].isin(ids)].copy()
    i = installments(ins, days_per_month, audit)
    b = bureau_features(buro, days_per_month, audit)
    z = balance_features(bb,b,audit)
    data = {'ins': (i,['late','dpd','underpaid','payment_ratio','shortfall','scheduled','paid']),
            'buro': (b,['credit','debt','overdue','overdue_days','debt_ratio','overdue_ratio','active']),
            'bb': (z,['delinquent','severity','unknown','closed'])}
    blocks = {'application':application_features(app)}
    # Each source uses the same windows and applicant index for fair ablations.
    for source,(x,metrics) in data.items():
        for label,bounds in WINDOWS.items():
            w = in_window(x,bounds)
            blocks[f'{source}_{label}'] = aggregate(w,metrics,f'{source}__{label}__',ids)
        blocks[f'{source}_dynamics'] = dynamics(x, ['late','dpd','underpaid'] if source=='ins' else ['delinquent','severity'] if source=='bb' else [],source,ids)
        # Compare separate periods, avoiding overlap between recent and old rates.
        for m in metrics:
            a=f'{source}__b0_3__{m}_mean'; old=f'{source}__b6_12__{m}_mean'
            if a in blocks[f'{source}_b0_3']:
                blocks[f'{source}_dynamics'][f'{source}__dyn__{m}_recent_minus_old'] = blocks[f'{source}_b0_3'][a]-blocks[f'{source}_b6_12'][old]
        # Longer-horizon dynamics use the same definitions as the original group.
        # Delta references are separate bins, never overlapping cumulative windows.
        for horizon, old_bin in [(24, 'b12_24'), (36, 'b24_36')]:
            dynamic_metrics = ['late','dpd','underpaid'] if source == 'ins' else ['delinquent','severity'] if source == 'bb' else []
            block = dynamics(x, dynamic_metrics, source, ids, horizon=horizon)
            for m in metrics:
                recent = f'{source}__b0_3__{m}_mean'
                old = f'{source}__{old_bin}__{m}_mean'
                if recent in blocks[f'{source}_b0_3']:
                    block[f'{source}__dyn{horizon}__{m}_recent_minus_old'] = blocks[f'{source}_b0_3'][recent] - blocks[f'{source}_{old_bin}'][old]
            blocks[f'{source}_dynamics{horizon}'] = block
    coverage = pd.DataFrame(index=ids)
    full = pd.concat(list(blocks.values()),axis=1).replace([np.inf,-np.inf], np.nan)
    if full.columns.duplicated().any() or not full.index.equals(ids): raise AssertionError('Unsafe aggregation')
    groups = {k:list(v.columns) for k,v in blocks.items()}
    experiments = experiment_columns(groups)
    return full,coverage,experiments,groups,audit

