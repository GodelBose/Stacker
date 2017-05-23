from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import helper
import re

class BaseModel:

    def __init__(self, name, params_path=''):
        '''Initializes a base model:
        -----------

        name: str
        Name of the model to be used. Follows a strict convention of TYPE-NAME
        where TYPE must be either "c" for classification or "r" for regression

        params_path: str
        Filepath to the parameter file that either sets the bounds for random parameter initializations or to load specific params


        Returns:
        --------
        BaseModel
        The Basemodel instance
        '''
        self.name = name
        self.__init_params__(params_path)

    def __init_params__(self, params_path):
        '''Initializes the parameters for scikit-learn models and stores them for xgboost:
        -----------

        params_path: str
        Filepath to the parameter file that either sets the bounds for random parameter initializations or to load specific params


        Returns:
        --------
        -
        '''
        f = open(params_path)
        txt = f.read()
        f.close()
        random_init = True if re.findall('min=',txt)!=[] else False
        if self.name == 'c-tree':
            print(0)
        elif self.name == 'c-rf':
            if random_init:
                self.params = helper.generate_rf_params(params_path)
            else:
                self.params = helper.read_params(params_path, 'rf')
        elif self.name == 'xgb':
            if random_init:
                self.params, self.num_rounds = helper.generate_xgb_params(params_path)
            else:
                self.params, self.num_rounds = helper.read_params(params_path, 'xgb')
        elif self.name == 'r-tree':
            print(0)
        elif self.name == 'r-rf':
            print(0)

    def fit(self, X, y):
        '''Trains the model:
        -----------

        X: numpy.array
        Data with axis n x m with "n" being the amount of instances and "m" the amount of features.

        y: numpy.array
        Labels corresponding to the data in X.


        Returns:
        --------
        -
        '''
        if 'xgb' in self.name:
            X_train = xgb.DMatrix(X, label=y)
            self.model = xgb.train(self.params, X_train, self.num_rounds)
        elif 'rf' in self.name:
            params = self.params
            model = RandomForestClassifier(n_estimators=params['n_estimators'], criterion='gini',
                                            max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'],
                                            max_leaf_nodes=params['max_leaf_nodes'],
                                            n_jobs=params['n_jobs'])
            model.fit(X,y)
            self.model = model

    def predict(self, X):
        '''Returns model predictions for the data matrix X:
        -----------

        X: numpy.array
        Data with axis n x m with "n" being the amount of instances and "m" the amount of features.


        Returns:
        --------
        predictions: numpy.array
        Predictions made by the wrapped model.
        '''
        if 'xgb' in self.name:
            X = xgb.DMatrix(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        #print(X.shape)
        if 'xgb' in self.name:
            self.predict(X)
        else:
            return self.model.predict_proba(X)
