import xgboost as xgb
import numpy as np
from base_model import BaseModel
import helper

class Stacker:

    def __init__(self, meta_model, base_models, num_splits, meta_model_params='', base_model_params=''):
        '''Initializes a meta stacking model:
        -----------

        meta_model: str
        Name of the meta model to be used. Follows a strict convention of TYPE-NAME
        where TYPE must be either "c" for classification or "r" for regression and NAME represents either
        xgb for XGBoost, rf for randomforest or dt for decisiontrees

        params_path: str
        Filepath to the parameter file that either sets the bounds for random parameter initializations or to load specific params

        num_splits: int
        The amount of splits that have to be created to create the meta training data.

        meta_model_params: str
        Path to the params file of the meta model

        base_model_params: str
        Path to the params directory of the base models

        Returns:
        --------
        -
        '''
        self.meta_model = BaseModel(meta_model, meta_model_params)
        self.base_models = [BaseModel(model, params) for model,params in zip(base_models, base_model_params)]
        self.num_splits = num_splits

    def generate_base_model_predictions(self, X, y):
        '''Split the training data and create predictions for each model to create the complete meta training data.
        -----------
        X: numpy.array
        Data matrix

        y:
        labels

        Returns:
        --------
        Meta training data that contains all base model predictions for each training instance
        '''
        model_predictions = []
        for model in self.base_models:
            split_predictions = []
            it  = helper.split_train_validation_data(X,y,self.num_splits)
            for X_train, y_train, X_valid, y_valid in it:
                model.fit(X_train, y_train)
                split_predictions.append(model.predict(X_valid))
            model_predictions.append(np.vstack(split_predictions))
        return np.hstack(model_predictions)

    def fit(self, X, y):
        '''Fit the meta model on the predictions made by all base models.
        -----------
        X: numpy.array
        Data matrix

        y:
        labels

        Returns:
        --------
        Meta training data that contains all base model predictions for each training instance
        '''
        X_train_meta = self.generate_base_model_predictions(X, y)
        self.meta_model.fit(X_train_meta, y)

    def predict(self, X, y):
        '''Fit the meta model on the predictions made by all base models.
        -----------
        X: numpy.array
        Data matrix

        y:
        labels

        Returns:
        --------
        Meta training data that contains all base model predictions for each training instance
        '''
        X_train_meta = self.generate_base_model_predictions(X, y)
        self.meta_model.predict(X_train_meta, y)
