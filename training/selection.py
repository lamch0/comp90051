"""Numeric redundancy screening learned on training rows only."""
import numpy as np

def select_numeric_training_columns(X_train, threshold=0.98, min_pairs=30):
    """Fit a deterministic numeric redundancy screen on TRAINING rows only.

    Returns (retained names, omission audit). Drop all-missing/constants and
    exact duplicates, then greedily omit |Pearson r| >= threshold using available
    pairs (at least min_pairs). Input order determines which column is preferred.
    Non-numeric categories pass through for downstream train-fitted encoding.
    This is unsupervised and does not use TARGET. It does not detect all linear
    combinations or nonlinear redundancy. Apply before encoding/imputation;
    also drop one level per categorical field and scale numeric features within
    the training fold. Missing-indicator redundancy needs a post-encoding check.
    """
    if not 0 < threshold <= 1 or min_pairs < 2:
        raise ValueError('threshold must be in (0,1], min_pairs >= 2')
    if X_train.columns.duplicated().any() or {'TARGET', 'SK_ID_CURR'} & set(X_train.columns):
        raise ValueError('Predictors must have unique names and exclude ID/TARGET')
    numeric = X_train.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    omitted, candidates = {}, []
    for c in numeric:
        if numeric[c].nunique(dropna=True) <= 1:
            omitted[c] = {'reason': 'all-missing or constant on training fold'}
        else:
            duplicate = next((other for other in candidates if numeric[c].equals(numeric[other])), None)
            if duplicate:
                omitted[c] = {'reason': 'exact duplicate on training fold', 'retained': duplicate}
            else:
                candidates.append(c)
    corr = numeric[candidates].corr(min_periods=min_pairs)
    retained_numeric = []
    for c in candidates:
        partner = next((other for other in retained_numeric if abs(corr.loc[c, other]) >= threshold), None)
        if partner:
            omitted[c] = {'reason': 'high absolute Pearson correlation on training fold',
                          'retained': partner, 'correlation': float(corr.loc[c, partner])}
        else:
            retained_numeric.append(c)
    return [c for c in X_train if c not in omitted], omitted
