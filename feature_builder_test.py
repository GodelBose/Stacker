import unittest
from feature_builder import FeatureBuilder
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
class TestFeatureBuilder(unittest.TestCase):

    def test_init(self):

        df = pd.DataFrame({'a':np.random.randint(0,3,10), 'b':np.random.randn(10), 'c':np.random.randn(10)})
        historical_df = df[:5]
        def create_f1(df):
            squares = df.a.apply(lambda x: x*2)
            return squares.values

        def create_f2(df, historical=historical_df):
            value_dict = historical[['a', 'b']].groupby('a').mean().to_dict()
            historical_feature = df.a.apply(lambda x: value_dict['b'].get(x,0))
            return historical_feature

        non_historical_features = [create_f1]
        historical_features = [create_f2]
        fb = FeatureBuilder(non_historical_features, historical_features)
        whole_dict = df[['a', 'b']].groupby('a').mean().to_dict()
        whole_dict_values = df.a.apply(lambda x: whole_dict['b'].get(x,0)).values
        self.assertTrue(len(fb.create_historical_features(df, historical_df))==len(df))
        self.assertTrue(len(fb.create_non_historical_features(df))==len(df))
        self.assertTrue((fb.create_non_historical_features(df)==df.a.values*2).sum() == len(df))
        self.assertTrue((fb.create_historical_features(df, historical_df)==whole_dict_values).sum() < len(df))

if __name__ == '__main__':
    unittest.main()
