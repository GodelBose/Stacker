import re
import numpy as np

'''
DecisionTreeClassifier(criterion='gini',
splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
min_impurity_split=1e-07, class_weight=None, presort=False)

RandomForestClassifier(n_estimators=10, criterion='gini',
 max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
  max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True,
   oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
'''


def read_param_file(params_path):
    '''Reads parameters from the given filepath and returns them in a dictionary
    -----------
    None

    Returns:

    params: dictionary
    Contains the parameters used for training the gradient boosting model.
    '''
    f = open(params_path,'r')
    txt = f.readline()
    param_dict = {}
    while txt!='':
        params = txt.split()
        param = re.sub('[^A-Za-z_]','',params[0])
        param_dict[param] = (param,'0','0')
        if param not in ['num_class', 'objective']:
            param_min = re.sub('[^0-9\.]','',params[1])
            param_max = re.sub('[^0-9\.]','',params[2])
            param_dict[param] = (param,param_min,param_max)
        else:
            param_value = re.sub('[^a-z:0-9]','', params[1])
            param_dict[param] = (param,param_value,'0')
        txt = f.readline()
    f.close()
    return param_dict

def generate_xgb_random_params(params_path,mode='c'):
    '''Generates random parameters for training a gradient boosting model.
    Parameters:
    -----------
    None

    Returns:

    params: dictionary
    Contains the parameters used for training the gradient boosting model.
    '''
    params_dict = read_param_file(params_path)
    params = {}
    # filling in default values
    params['max_depth'] = 6
    params['eta'] = 0.3
    params['min_child_weight'] = 1
    params['gamma'] = 0
    params['max_delta_step'] = 0
    params['colsample_bytree'] = 1
    params['colsample_bylevel'] = 1
    params['objective'] = 'reg:linear'
    params['subsample'] = 1
    params['silent'] = 1
    params['lambda'] = 1
    params['alpha'] = 0
    int_params = ['min_child_weight', 'max_depth']
    float_params = ['eta', 'colsample_bytree', 'colsample_bylevel', 'lambda', 'alpha', 'subsample', 'gamma']
    for key,value in params_dict.items():
        if key in int_params:
            params[key] = np.random.randint(int(value[1]),int(value[2]))
        elif key in float_params:
            params[key] = np.random.uniform(float(value[1]),float(value[2]))
        elif key == 'num_rounds':
            num_rounds = np.random.randint(int(value[1]),int(value[2]))
        elif key == 'num_class':
            params[key] = int(value[1])
        else:
            params[key] = value[1]
    return params, num_rounds
