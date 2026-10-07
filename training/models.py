"""Estimator factories. CV, tuning and scoring are implemented separately."""

def make_model(name, params, seed, threads):
    """Instantiate a fresh model; every candidate starts from scratch."""
    if name == 'lr':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(penalty='l2', solver='lbfgs', max_iter=2000,
                                  tol=1e-4, random_state=seed, **params)
    if name == 'cart':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(criterion='gini', min_samples_leaf=50,
                                      random_state=seed, **params)
    if name == 'lightgbm':
        from lightgbm import LGBMClassifier
        # Explicit GOSS and EFB connect the implementation to Ke et al. (2017).
        # No early stopping: the number of trees is fixed before outer evaluation.
        return LGBMClassifier(objective='binary', boosting_type='gbdt',
                              data_sample_strategy='goss', enable_bundle=True,
                              learning_rate=.05, n_estimators=300,
                              min_child_samples=50, reg_lambda=1.,
                              random_state=seed, n_jobs=threads,
                              deterministic=True, force_col_wise=True,
                              verbosity=-1, **params)
    raise ValueError(f'Unknown model: {name}')
