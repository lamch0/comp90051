"""Analytical metrics and synthetic integration tests; never load real data."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from .metrics import binary_metrics
from .preprocessing import FoldPreprocessor
from .run import run


def fixture(root):
    rng = np.random.default_rng(19)
    n = 600
    ids = pd.Index(np.arange(1000,1000+n), name='SK_ID_CURR')
    x = rng.normal(size=n)
    y = (x+rng.normal(size=n) > 1.1).astype(int)
    features = pd.DataFrame({'x':x, 'duplicate':x, 'noise':rng.normal(size=n),
                             'constant':1., 'category':np.where(x>0,'a','b')},index=ids)
    features.loc[ids[::7], 'noise'] = np.nan
    features.to_csv(root/'features_compact.csv')
    pd.Series(y,index=ids,name='TARGET').to_csv(root/'labels.csv')
    # Independent explicit stratified assignment; this tests consumption, not generation.
    folds = pd.DataFrame(index=ids)
    outer = np.zeros(n,dtype=int)
    for label in [0,1]:
        idx = np.flatnonzero(y == label)
        outer[idx] = np.arange(len(idx)) % 10
    folds['outer_fold'] = outer
    for k in range(10):
        inner = np.full(n,-1,dtype=int)
        for label in [0,1]:
            idx = np.flatnonzero((y == label)&(outer != k))
            inner[idx] = np.arange(len(idx)) % 3
        folds[f'inner_for_outer_{k}'] = inner
    folds.to_csv(root/'nested_folds.csv')
    (root/'experiments_compact.json').write_text(json.dumps({'E_test':list(features)}))
    return features, y, folds


def arguments(root, models):
    return SimpleNamespace(data_dir=str(root),output_dir=str(root/'results'),
                           grids=str(Path(__file__).with_name('grids.json')),
                           models=models, experiments=['E_test'], outer_folds=[0],
                           seed=42, threads=1, no_correlation_screen=False)


class MetricTests(unittest.TestCase):
    def test_perfect_reverse_and_ties(self):
        self.assertEqual(binary_metrics([0,1],[0.,1.])['roc_auc'],1.)
        self.assertEqual(binary_metrics([0,1],[1.,0.])['roc_auc'],0.)
        result = binary_metrics([0,1,0,1],[.3,.3,.3,.3])
        self.assertEqual(result['roc_auc'],.5)
        self.assertEqual(result['average_precision'],.5)
        self.assertEqual(result['tp'],0)
        self.assertEqual(result['fn'],2)

    def test_probability_and_classification_metrics(self):
        result = binary_metrics([0,1],[.25,.75])
        self.assertAlmostEqual(result['log_loss'],-np.log(.75))
        self.assertAlmostEqual(result['brier_score'],.0625)
        self.assertEqual(result['f1'],1.)
        # tied first score group: precision 1/2; next positive precision 2/3.
        self.assertAlmostEqual(binary_metrics([1,0,1],[.9,.9,.2])['average_precision'],7/12)

    def test_invalid_labels_not_silently_cast(self):
        with self.assertRaises(ValueError):binary_metrics([0,.5],[.1,.9])


class FoldTests(unittest.TestCase):
    def test_preprocessing_is_frozen(self):
        train = pd.DataFrame({'x':[1.,2.,np.nan],'cat':['a','b','a']})
        prep = FoldPreprocessor().fit(train)
        median = prep.numeric['x']['median']
        valid = pd.DataFrame({'x':[1000.,np.nan],'cat':['new','a']})
        transformed = prep.transform(valid)
        self.assertEqual(prep.numeric['x']['median'],median)
        self.assertEqual(transformed['cat__level_1'].sum(),0.)
        self.assertTrue(np.isfinite(transformed.to_numpy()).all())

    def test_nested_training_resume_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, folds = fixture(root)
            args = arguments(root,['lr','cart'])
            run(args)
            results = root/'results'
            for name in args.models:
                directory = results/'E_test'/name/'outer_0'
                self.assertEqual(len(list(directory.glob('candidate_*_inner_*.json'))),9)
                predictions = pd.read_csv(directory/'predictions.csv')
                self.assertEqual(set(predictions.SK_ID_CURR),set(folds.index[folds.outer_fold==0]))
                record = json.loads((directory/'result.json').read_text())
                self.assertEqual(record['preprocessing']['selected_raw_count'],3)
                self.assertTrue(0 <= record['metrics']['roc_auc'] <= 1)
            summary = pd.read_csv(results/'summary.csv')
            self.assertFalse(summary.complete.any())
            path = results/'E_test'/'lr'/'outer_0'/'result.json'
            before = path.stat().st_mtime_ns
            run(args)
            self.assertEqual(path.stat().st_mtime_ns,before)
            args.seed += 1
            with self.assertRaises(ValueError):run(args)


def lightgbm_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fixture(root)
        run(arguments(root,['lightgbm']))
        record = json.loads((root/'results/E_test/lightgbm/outer_0/result.json').read_text())
        assert 0 <= record['metrics']['roc_auc'] <= 1
        print('LightGBM synthetic smoke passed')


if __name__ == '__main__':
    import sys
    if '--lightgbm-smoke' in sys.argv:
        lightgbm_smoke()
    else:
        unittest.main()
