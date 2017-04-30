import unittest
import os
import helper

class TestHelper(unittest.TestCase):

    def test_params_reader(self):
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

    def test_random_params(self):
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
