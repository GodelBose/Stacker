import unittest
import os
import helper
import numpy as np

class TestHelper(unittest.TestCase):

    def test_params_reader_xgb(self):
        param_dict =helper.read_param_file('test_params')
        self.assertEqual(param_dict['eta'], ('eta','0.01', '0.2'))
        self.assertEqual(param_dict['gamma'], ('gamma','0', '1'))
        self.assertEqual(param_dict['max_depth'], ('max_depth','6', '23'))
        self.assertEqual(param_dict['min_child_weight'], ('min_child_weight','1', '20'))
        self.assertEqual(param_dict['subsample'], ('subsample','0.5', '0.98'))
        self.assertEqual(param_dict['colsample_bytree'], ('colsample_bytree','0.5', '1'))
        self.assertEqual(param_dict['colsample_bylevel'], ('colsample_bylevel','0.5', '1'))
        self.assertEqual(param_dict['lambda'], ('lambda','0', '1'))
        self.assertEqual(param_dict['alpha'], ('alpha','0', '1'))
        self.assertEqual(param_dict['num_class'], ('num_class','10', '0'))
        self.assertEqual(param_dict['objective'], ('objective','multi:softprob', '0'))

    def test_params_reader_rf(self):
        param_dict =helper.read_param_file('test_params_rf')
        self.assertEqual(param_dict['n_estimators'], ('n_estimators','10', '100'))
        self.assertEqual(param_dict['max_depth'], ('max_depth','3', '9'))
        self.assertEqual(param_dict['min_samples_split'], ('min_samples_split','2', '23'))
        self.assertEqual(param_dict['min_child_weight'], ('min_child_weight','1', '20'))
        self.assertEqual(param_dict['min_samples_leaf'], ('min_samples_leaf', '1', '111'))
        self.assertEqual(param_dict['max_leaf_nodes'], ('max_leaf_nodes','5', '100'))
        self.assertEqual(param_dict['min_impurity_split'], ('min_impurity_split','0.00001' , '0.001'))

    def test_random_params_xgb(self):
        param_dict, rounds = helper.generate_xgb_random_params('test_params')
        self.assertTrue(param_dict['eta']>=0.01)
        self.assertTrue(param_dict['eta']<=0.2)
        self.assertTrue(param_dict['gamma']>=0)
        self.assertTrue(param_dict['gamma']<=1)
        self.assertTrue(param_dict['max_depth']>=6)
        self.assertTrue(param_dict['max_depth']<=23)
        self.assertTrue(param_dict['min_child_weight']>=1)
        self.assertTrue(param_dict['min_child_weight']<=20)
        self.assertTrue(param_dict['subsample']>=0.5)
        self.assertTrue(param_dict['subsample']<=1)
        self.assertTrue(param_dict['colsample_bytree']>=0.5)
        self.assertTrue(param_dict['colsample_bytree']<=1)
        self.assertTrue(param_dict['colsample_bylevel']>=0.5)
        self.assertTrue(param_dict['colsample_bylevel']<=1)
        self.assertTrue(param_dict['lambda']>=0)
        self.assertTrue(param_dict['lambda']<=1)
        self.assertTrue(param_dict['alpha']>=0)
        self.assertTrue(param_dict['alpha']<=1)
        self.assertTrue(param_dict['num_class']==10)
        self.assertTrue(param_dict['objective']=='multi:softprob')

    def test_random_params_rf(self):
        param_dict =helper.generate_rf_random_params('test_params_rf')
        self.assertTrue(param_dict['n_estimators']>=10)
        self.assertTrue(param_dict['n_estimators']<100)
        self.assertTrue(param_dict['max_depth']>=3)
        self.assertTrue(param_dict['max_depth']<=9)
        self.assertTrue(param_dict['min_samples_split']>=2)
        self.assertTrue(param_dict['min_samples_split']<=23)
        self.assertTrue(param_dict['min_samples_leaf']>=1)
        self.assertTrue(param_dict['min_samples_leaf']<=111)
        self.assertTrue(param_dict['max_leaf_nodes']>=5)
        self.assertTrue(param_dict['max_leaf_nodes']<=100)
        self.assertTrue(param_dict['min_impurity_split']>=0.00001)
        self.assertTrue(param_dict['min_impurity_split']<=0.001)
        self.assertTrue(param_dict['n_jobs']==-1)



    def test_validation_generator(self):
        X,y = (np.random.randn(100,3), np.random.randint(0,2,100))
        it  = helper.split_train_validation_data(X,y,10)
        indices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        i=0
        for X_train, y_train, X_valid, y_valid, _, _ in it:
            if i < len(indices)-2:
                self.assertTrue((X_valid==X[indices[i]:indices[i+1]]).sum()==30)
            else:
                self.assertTrue((X_valid==X[indices[-2]:]).sum()==30)
            #print(X[indices[i]:indices[i+1]])
            self.assertTrue(X_train.shape[0]==90)
            self.assertTrue(y_train.shape[0]==90)
            self.assertTrue(len(X_valid)==10)
            self.assertTrue(len(y_valid)==10)
            i += 1

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
