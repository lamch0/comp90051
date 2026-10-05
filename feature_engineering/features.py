"""Current-application and historical record-level feature construction."""
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Fixed allowlist transcribed from the user-highlighted field guide.
APPLICATION_COLUMNS = tuple(json.loads((Path(__file__).with_name("application_columns.json")).read_text())["columns"])
from .schema import ID, KEY
from .cleaning import clean_application, clean_installments, clean_bureau, clean_balance

def divide(a, b):
    """Divide aligned pandas Series, treating a zero denominator as missing.

    For example, credit/income is undefined when recorded income is zero.
    NaN is retained for later imputation within a modelling training fold."""
    return a / b.where(b != 0)

def application_features(df):
    """Construct the current-application baseline (E0).

    Input: application_train, one row per SK_ID_CURR, including TARGET.
    Output: applicant-indexed original predictors plus financial ratios,
    EXT_SOURCE summaries and contact/document counts; TARGET is removed.
    Cleaning uses fixed semantic rules, not statistics estimated across applicants.
    Retains only highlighted original predictors from application_columns.json.
    Missing optional fields are tolerated for reduced-schema test fixtures."""
    cleaned = clean_application(df).set_index(ID).drop(columns='TARGET')
    x = cleaned[[c for c in APPLICATION_COLUMNS if c in cleaned]].copy()
    for name, a, b in [('CREDIT_INCOME_RATIO','AMT_CREDIT','AMT_INCOME_TOTAL'),
                       ('CREDIT_ANNUITY_RATIO','AMT_CREDIT','AMT_ANNUITY'),
                       ('GOODS_CREDIT_RATIO','AMT_GOODS_PRICE','AMT_CREDIT'),
                       ('INCOME_PER_FAMILY_MEMBER','AMT_INCOME_TOTAL','CNT_FAM_MEMBERS'),
                       ('CHILDREN_FAMILY_RATIO','CNT_CHILDREN','CNT_FAM_MEMBERS'),
                       ('EMPLOYED_TO_AGE_RATIO','DAYS_EMPLOYED','DAYS_BIRTH')]:
        if a in x and b in x: x[name] = divide(x[a], x[b])
    cols = [c for c in x if c.startswith('EXT_SOURCE_')]
    if cols:
        x['EXT_SOURCE_MEAN'] = x[cols].mean(axis=1)
        x['EXT_SOURCE_STD'] = x[cols].std(axis=1, ddof=0)
        x['EXT_SOURCE_N_MISSING'] = x[cols].isna().sum(axis=1)
        x['EXT_SOURCE_PRODUCT'] = x[cols].prod(axis=1, min_count=1)
    for name, cols in [('CONTACT_FLAGS_SUM', ['FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL']),
                       ('DOCUMENT_FLAGS_SUM', [c for c in x if c.startswith('FLAG_DOCUMENT_')])]:
        cols = [c for c in cols if c in x]
        if cols: x[name] = x[cols].sum(axis=1, min_count=1)
    return x

def installments(raw, days_per_month, audit):
    """Convert payment transactions into one record per scheduled installment.

    Input: installments_payments; KEY identifies a loan installment/version.
    days_per_month converts scheduled-date age from days to approximate months.
    audit is updated in place with excluded schedules and masked payment counts.

    Only schedules due by application day 0 are included. Payments after day 0
    or with unknown dates contribute neither amount nor completion information.
    Split payments are summed; completion is the first date the cumulative amount
    covers the schedule. Unsettled supplied schedules accrue DPD through day 0.
    Output includes age, paid/scheduled amounts, shortfall, underpaid, ratio,
    DPD (days past due) and late indicators. No applicant aggregation occurs here."""
    raw = clean_installments(raw, audit)
    observed = raw.observed_date.notna()
    x = raw.groupby(KEY, as_index=False).agg(due=('DAYS_INSTALMENT','first'),
        scheduled=('AMT_INSTALMENT','first'), paid=('observed_amount','sum'),
        last_payment=('observed_date','max'), payment_rows=('observed_date','count'),
        payment_information_unknown=('payment_information_unknown','max'))
    # No observed payment on a supplied schedule means zero observed paid, not future payment.
    x['paid'] = x.paid.fillna(0).where(~x.payment_information_unknown)
    audit['installments_unknown_payment_total'] = int(x.payment_information_unknown.sum())
    # Behaviour is undefined if the scheduled obligation is unknown/nonpositive.
    valid = x.scheduled.notna() & (x.scheduled > 0) & x.paid.notna()
    x['shortfall'] = (x.scheduled - x.paid).clip(lower=0).where(valid)
    # Tiny tolerance avoids labelling floating-point rounding as underpayment.
    x['underpaid'] = (x.paid + 1e-6 < x.scheduled).astype(float).where(valid)
    x['payment_ratio'] = divide(x.paid, x.scheduled).where(valid)
    # First date cumulative observed payments meet the schedule, not last overpayment.
    payments = raw.loc[observed].sort_values('observed_date').copy()
    payments['cumulative'] = payments.groupby(KEY).observed_amount.cumsum()
    completed = payments.loc[payments.cumulative + 1e-6 >= payments.AMT_INSTALMENT]
    completion = completed.groupby(KEY).observed_date.min().rename('completion')
    x = x.merge(completion, on=KEY, how='left', validate='one_to_one')
    # Underpaid installments are overdue through the application cutoff.
    endpoint = x.completion.where(x.underpaid == 0, 0)
    x['dpd'] = (endpoint - x.due).clip(lower=0).where(valid)
    x['late'] = (x.dpd > 0).astype(float).where(x.dpd.notna())
    x['age'] = -x.due / days_per_month
    return x

def bureau_features(raw, days_per_month, audit):
    """Prepare one record per external credit account from bureau.csv.

    Enforce unique account IDs, exclude future/unknown openings and positive
    update dates, and add credit/debt/overdue metrics and safe ratios when present.
    age refers to account OPENING, not the date of a repayment or debt snapshot.
    Current debt on an old account cannot reconstruct debt at its opening date.
    Output retains SK_ID_BUREAU so monthly balance rows can map to applicants.
    audit is updated in place with the excluded account count."""
    x = clean_bureau(raw, audit)
    x['age'] = -x.DAYS_CREDIT / days_per_month
    for output, source in [('credit','AMT_CREDIT_SUM'),('debt','AMT_CREDIT_SUM_DEBT'),
                            ('overdue','AMT_CREDIT_SUM_OVERDUE'),('overdue_days','CREDIT_DAY_OVERDUE')]:
        if source in x: x[output] = x[source]
    if 'credit' in x and 'debt' in x: x['debt_ratio'] = divide(x.debt,x.credit)
    if 'credit' in x and 'overdue' in x: x['overdue_ratio'] = divide(x.overdue,x.credit)
    if 'CREDIT_ACTIVE' in x:
        x['active'] = (x.CREDIT_ACTIVE == 'Active').astype(float).where(x.CREDIT_ACTIVE.notna())
    return x

def balance_features(raw, bureau, audit):
    """Map external account-month statuses to applicants through bureau.

    Input: bureau_balance and the eligible bureau account table.
    Require unique (SK_ID_BUREAU, MONTHS_BALANCE) observations and known codes.
    Exclude month 0 and future months; map -1 to age 0, -2 to age 1, etc.
    STATUS 0 means no delinquency; 1--5 are ordered severity categories.
    C=closed and X=unknown have separate flags and missing behavioural severity;
    neither is counted as an observed on-time repayment.
    Output: account-month rows with applicant ID, age and numeric status metrics.
    audit records nonhistorical and unmapped rows (counts may overlap)."""
    x = clean_balance(raw, bureau, audit)
    severity = pd.to_numeric(x.STATUS, errors='coerce')
    x['severity'] = severity
    x['delinquent'] = (severity > 0).astype(float).where(severity.notna())
    x['unknown'] = (x.STATUS == 'X').astype(float)
    x['closed'] = (x.STATUS == 'C').astype(float)
    # -1 is first historical month: age 0, -3 age 2, -4 age 3.
    x['age'] = -x.MONTHS_BALANCE - 1
    return x

