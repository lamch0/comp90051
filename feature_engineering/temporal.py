"""Window selection, applicant aggregation and monthly behavioural dynamics."""
import numpy as np
import pandas as pd
from .schema import ID
from .features import divide

def in_window(x, bounds):
    """Select records by age in a half-open interval [lo, hi).

    bounds=(0, 3) selects recent three-month records; (3, 6) selects the next bin.
    bounds=(0, None) selects all available eligible history. Exact age 3 belongs
    to [3, 6), so disjoint bins do not double-count boundary records."""
    lo, hi = bounds
    return x.loc[(x.age >= lo) & ((x.age < hi) if hi is not None else True)]

def aggregate(x, metrics, prefix, ids):
    """Summarise a source/window into one row for each applicant in ids.

    metrics names numeric behaviours or amounts; prefix identifies source/window.
    For each available metric, compute valid-value count, mean, max, sample std
    and sum. Also report total record count and has_history. For binary flags,
    mean is a rate and sum is the number of positive events.
    Missing history has count=0/has_history=0 and undefined metric statistics.
    Metric count excludes NaN, unlike total record count. Supplied ids preserves
    the same applicant cohort and ordering across all experimental feature sets."""
    g = x.groupby(ID)
    result = pd.DataFrame(index=ids)
    result[f'{prefix}count'] = g.size().reindex(ids, fill_value=0)
    result[f'{prefix}has_history'] = (result[f'{prefix}count'] > 0).astype('int8')
    for metric in metrics:
        if metric not in x: continue
        stats = g[metric].agg(['count','mean','max','std'])
        stats['sum'] = g[metric].sum(min_count=1)
        stats.columns = [f'{prefix}{metric}_{s}' for s in stats.columns]
        result = result.join(stats)
    # Reliability: distinguish one known observation among many unknown rows
    # from fully observed behaviour. No history remains NaN rather than 0.
    behaviour = 'late' if prefix.startswith('ins__') else 'delinquent' if prefix.startswith('bb__') else None
    if behaviour and behaviour in x:
        result[f'{prefix}valid_behavior_fraction'] = divide(g[behaviour].count(), g.size())
    return result

def monthly(x, metrics):
    """Reduce records to one mean per applicant and approximate month.

    floor(age) assigns a month bin. Within a month, events/accounts receive equal
    weight. Later dynamics give each observed monthly mean equal weight so a
    busy month does not dominate the slope. Absent months are not invented."""
    x = x.copy(); x['month'] = np.floor(x.age).astype(int)
    return x.groupby([ID,'month'], as_index=False)[metrics].mean()

def dynamics(x, metrics, source, ids, horizon=12):
    """Construct recency, monthly trends, volatility and delinquency streaks.

    Restrict to [0, horizon) months (default 12; also supports 24 and 36). Recency is the smallest observed age, with a
    separate recency for positive risky metrics. Trends use monthly means and
    ordinary least-squares slope against time=-month; positive means rising risk
    toward application. Sample std measures variation of monthly means.
    Slopes need two valid months. Streaks count consecutive applicant-months
    with any delinquency across accounts; gaps and unknown-only months break runs.
    Recent-minus-old deltas are added by build(), using disjoint-bin aggregates.
    Output is indexed by every applicant in ids; undefined dynamics remain NaN."""
    prefix = "dyn" if horizon == 12 else f"dyn{horizon}"
    x = in_window(x, (0,horizon))
    result = pd.DataFrame(index=ids)
    result[f'{source}__{prefix}__recency_months'] = x.groupby(ID).age.min()
    for m in metrics:
        if m not in x: continue
        risky = x.loc[x[m] > 0]
        result[f'{source}__{prefix}__{m}_recency_months'] = risky.groupby(ID).age.min()
    ms = [m for m in metrics if m in x]
    series = monthly(x, ms)
    # OLS slope = (n*sum(t*y) - sum(t)*sum(y)) /
    #             (n*sum(t*t) - sum(t)**2); no population-level fitting occurs.
    # Positive slope means worsening towards application (time = -age).
    for m in ms:
        v = series[[ID,'month',m]].dropna().copy()
        v['t'] = -v.month.astype(float); v['tt'] = v.t**2; v['ty'] = v.t*v[m]
        g = v.groupby(ID)
        n = g.size(); st = g.t.sum(); sy = g[m].sum()
        den = n*g.tt.sum()-st**2
        slope = divide(n*g.ty.sum()-st*sy, den).where(n >= 2)
        result[f'{source}__{prefix}__{m}_slope_per_month'] = slope
        result[f'{source}__{prefix}__{m}_monthly_std'] = g[m].std()
        result[f'{source}__{prefix}__{m}_observed_months'] = n.reindex(ids,fill_value=0)
    flag = 'late' if source == 'ins' else 'delinquent'
    if flag in x:
        # Applicant-month any delinquency; gaps and unknown-only months break runs.
        s = x.assign(month=np.floor(x.age).astype(int)).groupby([ID,'month'])[flag].max().reset_index()
        s = s.sort_values([ID,'month'], ascending=[True,False])
        previous = s.groupby(ID).month.shift()
        bad = s[flag].eq(1)
        start = (~bad) | previous.isna() | ((previous-s.month) != 1)
        s['run'] = start.groupby(s[ID]).cumsum()
        lengths = s.loc[bad].groupby([ID,'run']).size().groupby(ID).max()
        result[f'{source}__{prefix}__max_delinquency_streak_months'] = lengths.reindex(ids).where(ids.isin(s.loc[s[flag].notna(), ID]))
        result.loc[result.index.isin(s.loc[s[flag].notna(), ID]) & result.iloc[:,-1].isna(), result.columns[-1]] = 0
    return result

