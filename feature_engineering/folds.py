"""Optional reproducible nested-fold assignments; no model fitting."""
import numpy as np
import pandas as pd

def stratified_folds(labels, n_folds, seed):
    """Assign each binary-labelled applicant once to a balanced class-stratified fold."""
    y=np.asarray(labels)
    if n_folds < 2 or not np.isin(y,[0,1]).all():raise ValueError('Binary labels and at least two folds required')
    if min(np.sum(y==0),np.sum(y==1)) < n_folds:raise ValueError('Not enough examples in each class')
    rng=np.random.default_rng(seed);fold=np.empty(len(y),dtype=np.int16)
    for label in [0,1]:
        indices=np.flatnonzero(y==label);rng.shuffle(indices)
        fold[indices]=np.arange(len(indices)) % n_folds
    return fold


def nested_fold_plan(labels, outer=10, inner=3, seed=42):
    """Return one shared applicant-indexed plan; -1 means held-out outer test.

    inner_for_outer_k assigns ONLY the outer training applicants. During tuning,
    select one inner value as validation and the remaining inner values as train.
    Fit feature filtering, imputation, category vocabularies and scaling anew on
    that inner training subset. Refit on outer train after tuning, never outer test.
    """
    if not labels.index.is_unique:raise ValueError('Duplicate applicant IDs')
    result=pd.DataFrame(index=labels.index)
    result['outer_fold']=stratified_folds(labels,outer,seed)
    for fold in range(outer):
        train=result.outer_fold != fold
        result[f'inner_for_outer_{fold}']=-1
        result.loc[train,f'inner_for_outer_{fold}']=stratified_folds(labels.loc[train],inner,seed+fold+1)
    return result

