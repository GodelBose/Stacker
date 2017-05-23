import unittest
import os
import helper
from base_model import BaseModel
from sklearn.datasets import make_classification

n_samples = 5000
n_features = 15
num_classes = 10
X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5, n_redundant=3, n_classes=num_classes)

class TestHelper(unittest.TestCase):

    def test_init_xgb(self):
        model = BaseModel('xgb', 'test_params_random')
        self.assertEqual(model.name, 'xgb')
        self.assertEqual(model.params['num_class'], 10)
        self.assertTrue(model.num_rounds>=10)
        self.assertTrue(model.num_rounds<=75)

    def test_init_rf(self):
        model = BaseModel('c-rf', 'test_params_rf_random')
        self.assertEqual(model.name, 'c-rf')
        self.assertEqual(model.params['n_jobs'], -1)

    def test_fit_xgb(self):
        model = BaseModel('xgb', 'test_params_random')
        model.fit(X,y)
        self.assertEqual(model.predict(X).shape[0], n_samples)
        self.assertEqual(model.predict(X).shape[1], num_classes)

    def test_fit_rf(self):
        model = BaseModel('c-rf', 'test_params_rf_random')
        model.fit(X,y)
        self.assertEqual(model.predict_proba(X).shape[0], n_samples)
        self.assertEqual(model.predict_proba(X).shape[1], num_classes)

    def test_fixed_init_xgb(self):
        model = BaseModel('xgb', 'test_params_set')
        model.fit(X,y)
        self.assertEqual(model.predict(X).shape[0], n_samples)
        self.assertEqual(model.predict(X).shape[1],num_classes)

    def test_fixed_init_rf(self):
        model = BaseModel('c-rf', 'test_params_rf_set')
        model.fit(X,y)
        self.assertEqual(model.predict_proba(X).shape[0], n_samples)
        self.assertEqual(model.predict_proba(X).shape[1], num_classes)

if __name__ == '__main__':
    # write file for random xgboost model
    f = open('test_params_random','w')
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
    # write file for random random forest model
    f = open('test_params_rf_random', 'w')
    txt = '''n_estimators: min=10 max=100
max_depth: min=3 max=9
min_samples_split: min=2 max=23
min_child_weight: min=1 max=20
min_samples_leaf: min=1 max=111
max_leaf_nodes: min=5 max=100
min_impurity_split: min=0.00001 max=0.001'''
    f.write(txt)
    f.close()

    f = open('test_params_set','w')
    txt = '''eta: 0.03
    gamma: 0
    max_depth: 9
    min_child_weight: 4
    subsample: 0.67
    colsample_bytree: 0.88
    colsample_bylevel: 0.98
    lambda: 0.68
    alpha: 0
    num_class: 10
    objective: multi-softprob
    num_rounds: 23
    silent: 1'''
    f.write(txt)
    f.close()
    f = open('test_params_rf_set', 'w')
    txt = '''n_estimators: 55
max_depth: 6
min_samples_split: 7
min_child_weight: 3
min_samples_leaf: 3
max_leaf_nodes: 66
min_impurity_split: 0.00001
n_jobs: -1'''
    f.write(txt)
    f.close()
    unittest.main()
