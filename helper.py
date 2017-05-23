import re
import numpy as np

def split_train_validation_data(X, y, num_splits):
    '''Iterator to create train validation pairs for data and the corresponding labels
    -----------
    X: numpy.array
    Data matrix

    y:
    labels

    validation_percentage: float
    Size of each single validation set that is yielded by the iterator as percentage.

    Returns:
    --------
    train_X: numpy.array
    Training data of the split.

    train_y: numpy.array
    Training labels of the split.

    validation_X: numpy.array
    Validation data of the split.

    validation_y: numpy.array
    Validation labels of the split
    '''

    indices = list(np.arange(0, len(X), np.round(len(X)/num_splits)).astype(np.int32))
    for i in range(len(indices)):
        if i == len(indices)-1:
            yield X[:indices[-1]], y[:indices[-1]], X[indices[-1]:],  y[indices[-1]:], indices[-1], len(y)
        else:
            train_X = np.vstack([X[:indices[i]], X[indices[i+1]:]])
            validation_X = X[indices[i]:indices[i+1]]
            train_y = np.concatenate([y[:indices[i]], y[indices[i+1]:]], axis=0)
            validation_y = y[indices[i]:indices[i+1]]
            yield train_X, train_y, validation_X, validation_y, indices[i], indices[i+1]

def read_param_file(params_path):
    '''Reads parameters from the given filepath and returns them in a dictionary
    -----------
    params_path:string
    Path to the file containing the parameter data.

    Returns:
    --------
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

def generate_xgb_params(params_path,mode='c'):
    '''Generates random parameters for training a gradient boosting model.
    Parameters:
    -----------
    params_path:string
    Path to the file containing the parameter data.

    Returns:
    --------
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

def generate_rf_params(params_path,mode='c'):
    '''Generates random parameters for training a random forest model.
    Parameters:
    -----------
    params_path:string
    Path to the file containing the parameter data.

    Returns:
    --------
    params: dictionary
    Contains the parameters used for training the gradient boosting model.
    '''
    params_dict = read_param_file(params_path)
    params = {}
    # filling in default values
    params['n_estimators'] = 10
    params['max_depth'] = 9
    params['min_samples_split'] = 2
    params['min_samples_leaf'] = 1
    params['max_leaf_nodes'] = 0
    params['min_impurity_split'] = 1e-7
    params['n_jobs'] = -1
    int_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes']
    float_params = ['min_impurity_split']
    for key,value in params_dict.items():
        if key in int_params:
            params[key] = np.random.randint(int(value[1]),int(value[2]))
        elif key in float_params:
            params[key] = np.random.uniform(float(value[1]),float(value[2]))
    return params

def read_params(params_path, mode):
    '''Reads already set parameter values for a given model from the given filepath.
    Parameters:
    -----------
    params_path:string
    Path to the file containing the parameter data.

    Returns:
    --------
    params: dictionary
    Contains the parameters used for initializing the model.
    '''
    f = open(params_path,'r')
    txt = f.readline()
    if mode =='xgb':
        int_params = ['min_child_weight', 'max_depth', 'num_class', 'num_rounds', 'silent']
        float_params = ['eta', 'colsample_bytree', 'colsample_bylevel', 'lambda', 'alpha', 'subsample', 'gamma']
    elif mode=='rf':
        int_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes', 'n_jobs']
        float_params = ['min_impurity_split']
    param_dict = {}
    while txt!='':
        param_data = txt.split(':')
        param = re.sub('[^A-Za-z_]','',re.sub(' ','',param_data[0]))
        if param in int_params + float_params:
            param_dict[param] = int(re.sub('[^0-9\.]','',txt)) if param in int_params else float(re.sub('[^0-9\.]','',txt))
        else:
            param_dict[param] = re.sub('-',':',re.sub('[ \\n]','',param_data[1]))
        txt = f.readline()
    f.close()
    if mode =='xgb':
        return param_dict, int(param_dict['num_rounds'])
    else:
        return param_dict
