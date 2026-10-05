"""Fixed semantic feature simplification.

Fixed rules use column definitions, never labels or dataset statistics. Empirical
selection must be fitted separately on each training fold and then applied with
exactly the same retained names to its validation/test fold.
"""

BINARY = {'late', 'underpaid', 'active', 'delinquent', 'unknown', 'closed'}

# Recommended domain-led profile: each retained statistic answers a distinct
# question. These are hypotheses chosen before evaluation, not proven winners.
INTERPRETABLE_AGGREGATES = {
    'ins': {'count', 'late_mean', 'dpd_mean', 'dpd_max', 'underpaid_mean',
            'payment_ratio_mean', 'shortfall_mean', 'valid_behavior_fraction'},
    'buro': {'count', 'credit_mean', 'debt_ratio_mean', 'overdue_ratio_mean',
             'overdue_days_max', 'active_mean'},
    'bb': {'count', 'delinquent_mean', 'severity_max', 'unknown_mean', 'valid_behavior_fraction'},
}
INTERPRETABLE_DYNAMICS = {
    'ins': {'recency_months', 'dpd_recency_months', 'dpd_slope_per_month',
            'dpd_monthly_std', 'max_delinquency_streak_months',
            'late_recent_minus_old', 'payment_ratio_recent_minus_old'},
    # Current snapshots across opening cohorts cannot measure repayment change.
    'buro': {'recency_months'},
    'bb': {'recency_months', 'delinquent_recency_months', 'severity_slope_per_month',
           'severity_monthly_std', 'max_delinquency_streak_months',
           'delinquent_recent_minus_old'},
}

def interpretability_reason(column):
    """Omit secondary statistics from the recommended explainable profile."""
    parts = column.split('__')
    if len(parts) != 3:
        return None
    source, representation, metric = parts
    candidates = INTERPRETABLE_DYNAMICS if representation.startswith('dyn') else INTERPRETABLE_AGGREGATES
    if metric not in candidates[source]:
        return 'Secondary statistic omitted; retain representative frequency, severity, burden or dynamics'
    return None

def semantic_reason(column):
    """Return why a column is omitted from the compact profile, or None.

    Keep binary rates, numeric mean/max/std and one record count per block.
    This removes redundant encodings without collapsing research time windows.
    Raw housing AVG values represent each family; MODE/MEDI are omitted when
    building profiles only if their corresponding AVG column is present.
    """
    if column in {'CONTACT_FLAGS_SUM', 'DOCUMENT_FLAGS_SUM'}:
        return 'Sum of retained original binary flags'
    if column in {'EXT_SOURCE_MEAN', 'EXT_SOURCE_PRODUCT'}:
        return 'Retain original EXT_SOURCE scores plus dispersion/missingness'
    parts = column.split('__')
    if len(parts) != 3:
        return None
    source, representation, metric = parts
    if representation.startswith('dyn'):
        if source == 'bb' and metric == 'severity_recency_months':
            return 'Same positive-event recency as delinquent status'
        if metric == 'late_recency_months':
            return 'Same positive-event recency as dpd'
        if metric.endswith('_observed_months'):
            preferred = 'dpd_observed_months' if source == 'ins' else 'severity_observed_months'
            if metric != preferred:
                return 'Same valid monthly support as retained behavioural metric'
        return None
    if metric == 'has_history':
        return 'Deterministic indicator of retained record count > 0'
    if metric in {'count', 'valid_behavior_fraction'}:
        return None
    name, stat = metric.rsplit('_', 1)
    if stat == 'count':
        return 'Keep record count; metric-specific missing support can be checked separately'
    if name in BINARY and stat != 'mean':
        return 'Keep binary rate; omit related count/sum/max/std representations'
    if name not in BINARY and stat == 'sum':
        return 'Keep mean/max/std; sum combines observation volume and magnitude'
    return None

def compact_experiments(experiments):
    """Return domain-led compact lists and a per-experiment omission audit.

    Preserve eight application engineered features, 19 aggregate feature types
    per window and 14 dynamics types per horizon (41 engineered types total).

    Original experiment names and temporal scale membership are preserved.
    Existing full feature matrices remain compatible: select these column lists.
    """
    profiles, audit = {}, {}
    for experiment, columns in experiments.items():
        omitted = {}
        for column in columns:
            reason = semantic_reason(column) or interpretability_reason(column)
            if column.endswith(('_MODE', '_MEDI')) and column.rsplit('_', 1)[0] + '_AVG' in columns:
                reason = 'Use AVG as the single housing-summary representation'
            if reason:
                omitted[column] = reason
        profiles[experiment] = [c for c in columns if c not in omitted]
        audit[experiment] = {'input_count': len(columns), 'retained_count': len(profiles[experiment]),
                             'omitted': omitted}
    return profiles, audit
