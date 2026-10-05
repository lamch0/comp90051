"""Fixed data cleaning, schema validation and as-of-application safeguards.

Population-dependent preprocessing (imputation, scaling, learned clipping and
encoding) must still be fitted separately inside each modelling training fold.
All functions return copies and never modify the source CSV files.
"""
import numpy as np
import pandas as pd
from .schema import ID, KEY, REQUIRED

def require(df, columns, name):
    """Validate required columns and the first (primary) key column.

    Other composite-key checks belong to the table-specific preparation functions.
    Raise ValueError instead of silently building features from an invalid schema."""
    missing = set(columns) - set(df.columns)
    if missing: raise ValueError(f'{name}: missing columns {sorted(missing)}')
    if df[columns[:1]].isna().any().any(): raise ValueError(f'{name}: null key')


def clean_application(df):
    """Copy the application table and replace known sentinel/unknown values.

    This uses fixed rules; it does not impute, clip quantiles or learn categories.
    Retain SK_ID_CURR and TARGET so the caller can manage labels separately.
    """
    x = df.copy()
    if 'DAYS_EMPLOYED' in x: x['DAYS_EMPLOYED'] = x.DAYS_EMPLOYED.replace(365243, np.nan)
    for c, v in [('CODE_GENDER', 'XNA'), ('NAME_FAMILY_STATUS', 'Unknown'), ('ORGANIZATION_TYPE', 'XNA')]:
        if c in x: x[c] = x[c].replace(v, np.nan)
    return x


def clean_installments(raw, audit):
    """Validate schedules and mask payments unavailable at application day 0.

    Return eligible transaction rows with observed_amount/observed_date columns.
    Do not deduplicate apparently identical transactions: they can be real payments.
    Update audit in place; schedule consolidation belongs to feature construction.
    """
    require(raw, REQUIRED['installments_payments'], 'installments')
    if raw[KEY].isna().any().any(): raise ValueError('Null installment key')
    raw = raw.copy()
    # A schedule is observable only if it is due on/before the current application.
    eligible = raw.DAYS_INSTALMENT.notna() & (raw.DAYS_INSTALMENT <= 0)
    audit['installment_future_or_unknown_due_rows'] = int((~eligible).sum())
    raw = raw.loc[eligible].copy()
    conflicts = raw.groupby(KEY)[['DAYS_INSTALMENT','AMT_INSTALMENT']].nunique()
    if (conflicts > 1).any().any(): raise ValueError('Conflicting schedule within installment key')
    observed = raw.DAYS_ENTRY_PAYMENT.notna() & (raw.DAYS_ENTRY_PAYMENT <= 0)
    audit['payment_future_or_unknown_rows_masked'] = int((~observed).sum())
    if (raw.loc[observed, 'AMT_PAYMENT'] < 0).any(): raise ValueError('Negative payment amount')
    # An unknown transaction date or missing amount on a dated historical
    # payment prevents establishing the installment's paid total. Never turn
    # an unknown payment into evidence of zero payment/delinquency.
    raw['payment_information_unknown'] = raw.DAYS_ENTRY_PAYMENT.isna() | (observed & raw.AMT_PAYMENT.isna())
    audit['payment_information_unknown_rows'] = int(raw.payment_information_unknown.sum())
    raw['observed_amount'] = raw.AMT_PAYMENT.where(observed)
    raw['observed_date'] = raw.DAYS_ENTRY_PAYMENT.where(observed)
    return raw


def clean_bureau(raw, audit):
    """Validate account keys and exclude unavailable openings/updates.

    Return a copy of eligible accounts, retaining original columns. Update audit.
    """
    if raw.SK_ID_BUREAU.isna().any() or raw.SK_ID_BUREAU.duplicated().any():
        raise ValueError('bureau: SK_ID_BUREAU must be unique and non-null')
    keep = raw.DAYS_CREDIT.notna() & (raw.DAYS_CREDIT <= 0)
    if 'DAYS_CREDIT_UPDATE' in raw:
        keep &= raw.DAYS_CREDIT_UPDATE.isna() | (raw.DAYS_CREDIT_UPDATE <= 0)
    audit['bureau_future_or_unknown_open_rows'] = int((~keep).sum())
    x = raw.loc[keep].copy()
    return x


def clean_balance(raw, bureau, audit):
    """Validate monthly statuses, enforce pre-application months and map IDs.

    Return account-month rows linked to eligible applicants through bureau.
    Unmapped/nonhistorical counts can overlap; this function updates audit.
    """
    if raw[['SK_ID_BUREAU','MONTHS_BALANCE']].isna().any().any(): raise ValueError('Null balance key')
    if raw.duplicated(['SK_ID_BUREAU','MONTHS_BALANCE']).any(): raise ValueError('Duplicate account-month')
    allowed = {'0','1','2','3','4','5','C','X'}
    raw = raw.copy(); raw['STATUS'] = raw.STATUS.astype('string')
    if (~raw.STATUS.isin(allowed)).any(): raise ValueError('Unexpected/missing bureau STATUS')
    if (raw.MONTHS_BALANCE % 1 != 0).any(): raise ValueError('Noninteger balance month')
    # Exclude application month: its within-month availability is not documented.
    audit['balance_nonhistorical_rows'] = int((raw.MONTHS_BALANCE >= 0).sum())
    audit['balance_unmapped_rows'] = int((~raw.SK_ID_BUREAU.isin(bureau.SK_ID_BUREAU)).sum())
    x = raw.loc[raw.MONTHS_BALANCE < 0].merge(bureau[[ID,'SK_ID_BUREAU']], on='SK_ID_BUREAU', validate='many_to_one')
    return x
