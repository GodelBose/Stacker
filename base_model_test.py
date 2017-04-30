import unittest
import os
import helper
from base_model import BaseModel
from sklearn.datasets import make_classification

class TestHelper(unittest.TestCase):

    def test_init(self):
        model = BaseModel('xgb', 'test_params')
        self.assertEqual(model.name, 'xgb')
        self.assertEqual(model.params['num_class'], 10)
        self.assertTrue(model.num_rounds>=10)
        self.assertTrue(model.num_rounds<=75)

    def test_fit(self):
        n_samples = 5000
        n_features = 15
        num_classes = 10
        model = BaseModel('xgb', 'test_params')
        X,y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5, n_redundant=3, n_classes=num_classes)
        #print(X.shape)
        model.fit(X,y)
        self.assertEqual(model.predict(X).shape[0], n_samples)
        self.assertEqual(model.predict(X).shape[1], num_classes)

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
