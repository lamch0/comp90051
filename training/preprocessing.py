"""Population-fitted preprocessing for a future TRAINING fold, with no model.

Do not fit this on features_compact.csv as a whole. During nested CV, fit one
instance on inner train, transform inner validation; after tuning, fit a fresh
instance on outer train and transform outer test. Unknown numeric values are
represented by training median plus a missing indicator where train has missing.
"""
import numpy as np
import pandas as pd

class FoldPreprocessor:
    """Train-local median imputation, missing indicators, scaling and one-hot coding.

    Fit never receives labels. One reference category is omitted per field; unseen
    categories transform to all zeros (same encoding as reference), without
    extending vocabulary from validation/test. All-missing numeric columns are
    omitted. Constants after encoding are removed using training rows only.
    Correlation screening is a separate optional step before this class.
    """
    def fit(self, X_train):
        if X_train.columns.duplicated().any() or {'TARGET','SK_ID_CURR'} & set(X_train):
            raise ValueError('Unique predictors required; exclude ID/TARGET')
        self.input_columns=list(X_train.columns)
        self.numeric={};self.categories={};self.fit_rows=len(X_train)
        if not self.fit_rows:raise ValueError('Empty training fold')
        numeric=set(X_train.select_dtypes(include=[np.number]).columns)
        for c in self.input_columns:
            if c in numeric:
                values=X_train[c].astype(float).replace([np.inf,-np.inf],np.nan)
                if not values.notna().any():continue
                median=float(values.median());filled=values.fillna(median)
                scale=float(filled.std(ddof=0))
                self.numeric[c]={'median':median,'mean':float(filled.mean()),
                                 'scale':scale if scale>0 else 1.,'missing_indicator':bool(values.isna().any())}
            else:
                self.categories[c]=sorted(set(self._category(v) for v in X_train[c]))
        encoded=self._encode(X_train)
        self.output_columns=[c for c in encoded if encoded[c].nunique()>1]
        self.fitted=True
        return self

    @staticmethod
    def _category(value):
        # Tagged missing value cannot collide with any real category string.
        return (0,'') if pd.isna(value) else (1,str(value))

    def _encode(self, X):
        if list(X.columns)!=self.input_columns:raise ValueError('Apply identical selected columns in identical order')
        columns={}
        for c in self.input_columns:
            if c in self.numeric:
                params=self.numeric[c]
                values=X[c].astype(float).replace([np.inf,-np.inf],np.nan)
                columns[c]=(values.fillna(params['median'])-params['mean'])/params['scale']
                if params['missing_indicator']:columns[c+'__missing']=values.isna().astype(float)
            elif c in self.categories:
                values=X[c].map(self._category)
                # First train category is reference; a single-level field contributes none.
                for number,category in enumerate(self.categories[c][1:],start=1):
                    columns[f'{c}__level_{number}']=values.map(lambda value:float(value==category))
        return pd.DataFrame(columns,index=X.index)

    def transform(self, X):
        """Apply frozen training transformations without refitting on this input."""
        if not getattr(self,'fitted',False):raise ValueError('Fit on a training fold first')
        result=self._encode(X)[self.output_columns]
        if not np.isfinite(result.to_numpy(dtype=float)).all():raise ValueError('Non-finite encoded output')
        return result
