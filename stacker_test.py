import unittest
import os
import helper
from stacker import Stacker
from sklearn.datasets import make_classification
from feature_builder import FeatureBuilder
import pandas as pd
import numpy as np

class TestStacker(unittest.TestCase):

    def test_init_xgb(self):
        models = ['xgb' for x in range(5)]
        params = ['test_params' for x in range(5)]
        feature_builder = FeatureBuilder([], [])
        meta_clf = Stacker('xgb', models, 10, feature_builder, meta_model_params='test_params', base_model_params=params)
        self.assertEqual(meta_clf.meta_model.name, 'xgb')
        self.assertEqual(meta_clf.meta_model.params['num_class'], 10)
        self.assertTrue(meta_clf.meta_model.num_rounds>=10)
        self.assertTrue(meta_clf.meta_model.num_rounds<=75)

    def test_init_rf(self):
        models = ['c-rf' for x in range(5)]
        params = ['test_params_rf' for x in range(5)]
        feature_builder = FeatureBuilder([], [])
        meta_clf = Stacker('xgb', models, 10, feature_builder, meta_model_params='test_params', base_model_params=params)
        self.assertEqual(meta_clf.meta_model.name, 'xgb')
        self.assertEqual(meta_clf.meta_model.params['num_class'], 10)
        self.assertTrue(meta_clf.meta_model.num_rounds>=10)
        self.assertTrue(meta_clf.meta_model.num_rounds<=75)

    def test_base_predictions_xgb(self):
        n_samples = 5000
        n_features = 15
        num_classes = 10
        models = ['xgb' for x in range(5)]
        params = ['test_params' for x in range(5)]
        fb = FeatureBuilder([], [])
        meta_clf = Stacker('xgb', models, 10, fb, meta_model_params='test_params', base_model_params=params)
        self.assertEqual(meta_clf.meta_model.name, 'xgb')
        X,y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5, n_redundant=3, n_classes=num_classes)
        meta_prediction = meta_clf.generate_base_model_predictions(X,y)
        self.assertTrue(meta_prediction.shape[0]==n_samples)
        self.assertTrue(meta_prediction.shape[1]==num_classes*len(models))

    def test_stacker_historic_fit_rf(self):
        n_samples = 5000
        n_features = 15
        num_classes = 10
        models = ['c-rf' for x in range(5)]
        params = ['test_params_rf' for x in range(5)]
        X,y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5, n_redundant=3, n_classes=num_classes)
        df = pd.DataFrame({'feature_'+str(i):X[:,i] for i in range(X.shape[1])})
        df['ID'] = np.random.randint(0,100,n_samples)
        historical_df = df[:2500]

        def create_f1(df):
            squares = df.feature_2.apply(lambda x: x*2)
            return squares.values

        def create_f2(df):
            some_feature = df.feature_3.apply(lambda x: np.sin(x)/np.cos(x*2))
            return some_feature.values

        def create_historic_f1(df, historical=historical_df):
            value_dict = historical[['ID', 'feature_1']].groupby('ID').mean().to_dict()
            historical_feature = df.ID.apply(lambda x: value_dict['feature_1'].get(x,0))
            return historical_feature

        def create_historic_f2(df, historical=historical_df):
            value_dict = historical[['ID', 'feature_5']].groupby('ID').median().to_dict()
            historical_feature = df.ID.apply(lambda x: value_dict['feature_5'].get(x,0))
            return historical_feature

        non_historical_features = [create_f1, create_f2]
        historical_features = [create_historic_f1, create_historic_f2]
        fb = FeatureBuilder(non_historical_features, historical_features)
        meta_clf = Stacker('xgb', models, 10, fb, meta_model_params='test_params', base_model_params=params)
        meta_clf.fit(X, y, df)
        self.assertTrue(meta_clf.predict(X[-1000:], df=df[-1000:], historical_df=df).shape[0]==1000)

    def test_stacker_historic_fit_xgb(self):
        n_samples = 5000
        n_features = 15
        num_classes = 10
        models = ['xgb' for x in range(5)]
        params = ['test_params' for x in range(5)]
        X,y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5, n_redundant=3, n_classes=num_classes)
        df = pd.DataFrame({'feature_'+str(i):X[:,i] for i in range(X.shape[1])})
        df['ID'] = np.random.randint(0,100,n_samples)
        historical_df = df[:2500]

        def create_f1(df):
            squares = df.feature_2.apply(lambda x: x*2)
            return squares.values

        def create_f2(df):
            some_feature = df.feature_3.apply(lambda x: np.sin(x)/np.cos(x*2))
            return some_feature.values

        def create_historic_f1(df, historical=historical_df):
            value_dict = historical[['ID', 'feature_1']].groupby('ID').mean().to_dict()
            historical_feature = df.ID.apply(lambda x: value_dict['feature_1'].get(x,0))
            return historical_feature

        def create_historic_f2(df, historical=historical_df):
            value_dict = historical[['ID', 'feature_5']].groupby('ID').median().to_dict()
            historical_feature = df.ID.apply(lambda x: value_dict['feature_5'].get(x,0))
            return historical_feature

        non_historical_features = [create_f1, create_f2]
        historical_features = [create_historic_f1, create_historic_f2]
        fb = FeatureBuilder(non_historical_features, historical_features)
        meta_clf = Stacker('xgb', models, 10, fb, meta_model_params='test_params', base_model_params=params)
        meta_clf.fit(X, y, df)
        self.assertTrue(meta_clf.predict(X[-1000:], df=df[-1000:], historical_df=df).shape[0]==1000)

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
    f = open('test_params_rf', 'w')
    txt = '''n_estimators: min=10 max=100
max_depth: min=3 max=9
min_samples_split: min=2 max=23
min_child_weight: min=1 max=20
min_samples_leaf: min=1 max=111
max_leaf_nodes: min=5 max=100
min_impurity_split: min=0.00001 max=0.001'''
    f.write(txt)
    f.close()
    unittest.main()
