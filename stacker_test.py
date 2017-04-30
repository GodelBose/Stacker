import unittest
import os
import helper
from stacker import Stacker
from sklearn.datasets import make_classification

class TestStacker(unittest.TestCase):

    def test_init(self):
        models = ['xgb' for x in range(5)]
        params = ['test_params' for x in range(5)]
        meta_clf = Stacker('xgb', models, 10, meta_model_params='test_params', base_model_params=params)
        self.assertEqual(meta_clf.meta_model.name, 'xgb')
        self.assertEqual(meta_clf.meta_model.params['num_class'], 10)
        self.assertTrue(meta_clf.meta_model.num_rounds>=10)
        self.assertTrue(meta_clf.meta_model.num_rounds<=75)

    def test_base_predictions(self):
        n_samples = 5000
        n_features = 15
        num_classes = 10
        models = ['xgb' for x in range(5)]
        params = ['test_params' for x in range(5)]
        meta_clf = Stacker('xgb', models, 10, meta_model_params='test_params', base_model_params=params)
        self.assertEqual(meta_clf.meta_model.name, 'xgb')
        X,y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5, n_redundant=3, n_classes=num_classes)
        meta_prediction = meta_clf.generate_base_model_predictions(X,y)
        print(meta_prediction.shape)
        self.assertTrue(meta_prediction.shape[0]==n_samples)
        self.assertTrue(meta_prediction.shape[1]==num_classes*len(models))


if __name__ == '__main__':
    f = open('test_params','w')
    txt = '''eta: min=0.01 max=0.2
    gamma: min=0 max=1
    max_depth: min=6 max=23
    min_child_weight: min=1 max=20
    subsample: min=0.5 max=0.98
    colsample_bytree: min=0.5 max=1
    colsample_bylevel: min=0.5 max=1
    lambda: min=0 max=1
    alpha: min=0 max=1
    num_class: 10
    objective: multi:softprob
    num_rounds: min=10 max=75'''
    f.write(txt)
    f.close()

    unittest.main()
