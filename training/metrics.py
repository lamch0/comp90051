"""Binary metrics implemented with NumPy; no third-party scoring functions.

PR-AUC here means average precision (step-weighted precision), not trapezoidal
PR area. ROC uses groups of tied scores, so ties receive the correct half credit.
"""
import numpy as np


def binary_metrics(y, probability):
    """Return ranking, probability and fixed-threshold classification metrics."""
    y = np.asarray(y)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError('Nonempty aligned one-dimensional arrays required')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Binary labels and finite probabilities in [0,1] required')
    positives, negatives = int(y.sum()), int((1-y).sum())
    if not positives or not negatives:
        raise ValueError('Ranking evaluation requires both classes')
    order = np.argsort(-p, kind='stable')
    scores, labels = p[order], y[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(y)-1]
    tp_curve = np.cumsum(labels)[ends]
    fp_curve = ends+1-tp_curve
    tpr = np.r_[0., tp_curve/positives]
    fpr = np.r_[0., fp_curve/negatives]
    auc = float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2))
    recall_curve = tp_curve/positives
    precision_curve = tp_curve/(ends+1)
    ap = float(np.sum(np.diff(np.r_[0., recall_curve])*precision_curve))
    clipped = np.clip(p, 1e-15, 1-1e-15)
    prediction = p >= .5  # Prespecified threshold; never selected from outer test.
    tp = int(np.sum(prediction & (y == 1)))
    fp = int(np.sum(prediction & (y == 0)))
    fn = positives-tp
    tn = negatives-fp
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/positives
    return {'roc_auc': auc, 'average_precision': ap,
            'log_loss': float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))),
            'brier_score': float(np.mean((p-y)**2)),
            'accuracy': (tp+tn)/len(y), 'precision': precision, 'recall': recall,
            'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
