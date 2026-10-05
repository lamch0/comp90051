"""Shared applicant keys, input schemas and temporal window definitions."""

# Current loan application ID: the final modelling unit is one applicant row.
ID = 'SK_ID_CURR'
# Multiple payment transactions can belong to the same scheduled installment.
KEY = [ID, 'SK_ID_PREV', 'NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER']
# w = cumulative window; b = disjoint bin; None means no oldest-age cutoff.
WINDOWS = {'all': (0, None), 'w3': (0, 3), 'w6': (0, 6), 'w12': (0, 12), 'w24': (0, 24), 'w36': (0, 36),
           'b0_3': (0, 3), 'b3_6': (3, 6), 'b6_12': (6, 12), 'b12_24': (12, 24), 'b24_36': (24, 36)}
# Minimum schema; optional amount/status fields are used when supplied.
REQUIRED = {
 'application_train': [ID, 'TARGET'],
 'installments_payments': KEY + ['DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT'],
 'bureau': [ID, 'SK_ID_BUREAU', 'DAYS_CREDIT'],
 'bureau_balance': ['SK_ID_BUREAU', 'MONTHS_BALANCE', 'STATUS']}

