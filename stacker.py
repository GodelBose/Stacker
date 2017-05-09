import xgboost as xgb
import numpy as np
from base_model import BaseModel
import helper
from feature_builder import FeatureBuilder
import pandas as pd

class Stacker:

    def __init__(self, meta_model, base_models, num_splits, feature_builder, meta_model_params='', base_model_params=''):
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

        feature_builder: FeatureBuilder
        Instance of FeatureBuilder class already initialized with all feature functions.

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
        self.feature_builder = feature_builder

    def generate_base_model_predictions(self, X, y, df=None):
        '''Split the training data and create predictions for each model to create the complete meta training data.
        -----------
        X: numpy.array
        Data matrix

        y:
        labels

        df: pandas.DataFrame
        Raw DataFrame to be used if historical features need to be created.

        Returns:
        --------
        Meta training data that contains all base model predictions for each training instance
        '''
        model_predictions = []
        for model in self.base_models:
            is_xgb = True if 'xgb' in model.name else False
            is_c = True if 'c' == model.name[0] else False
            split_predictions = []
            it  = helper.split_train_validation_data(X,y,self.num_splits)
            for X_train, y_train, X_valid, y_valid, index1, index2 in it:
                if not isinstance(df, pd.DataFrame):
                    model.fit(X_train, y_train)
                    if is_xgb or not is_c:
                        split_predictions.append(model.predict(X_valid))
                    else:
                        split_predictions.append(model.predict_proba(X_valid))
                else:
                    historical_df = df[index1:index2]
                    df_temp = pd.concat([df[:index1], df[index2:]])
                    historical_features = self.feature_builder.create_historical_features(df, historical_df)
                    train_historical_features = np.concatenate([historical_features[:index1], historical_features[index2:]], axis=0)
                    validation_historical_features = historical_features[index1:index2]
                    X_train = np.hstack([X_train, train_historical_features])
                    X_valid = np.hstack([X_valid, validation_historical_features])
                    model.fit(X_train, y_train)
                    if is_xgb or not is_c:
                        split_predictions.append(model.predict(X_valid))
                    else:
                        split_predictions.append(model.predict_proba(X_valid))
            model_predictions.append(np.vstack(split_predictions))
        return np.hstack(model_predictions)

    def generate_new_base_model_predictions(self, X, df, historical_df):
        model_predictions = []
        for model in self.base_models:
            is_xgb = True if 'xgb' in model.name else False
            is_c = True if 'c' == model.name[0] else False
            if isinstance(df, pd.DataFrame):
                historical_features = self.feature_builder.create_historical_features(df, historical_df)
                X_temp = np.hstack([X, historical_features])
            if is_xgb or not is_c:
                model_predictions.append(model.predict(X_temp))
            else:
                model_predictions.append(model.predict_proba(X_temp))
        return np.hstack(model_predictions)

    def fit(self, X, y, df=None):
        '''Fit the meta model on the predictions made by all base models.
        -----------
        X: numpy.array
        Data matrix

        y: numpy.array
        labels

        df: pd.DataFrame
        Raw DataFrame to be used for creating the features that have to be created using historical knowledge.

        Returns:
        --------
        Meta training data that contains all base model predictions for each training instance
        '''
        X_train_meta = self.generate_base_model_predictions(X, y, df=df)
        self.meta_model.fit(X_train_meta, y)

    def predict(self, X, df=None, historical_df=None):
        '''Predict with the meta model on the predictions made by all base models.
        -----------
        X: numpy.array
        Data matrix

        df: pd.DataFrame
        Raw DataFrame to be used for creating the features that have to be created using historical knowledge.

        historical_df: pd.DataFrame
        Raw DataFrame containing the historical knowledge to create the predictions.

        Returns:
        --------
        Meta training data that contains all base model predictions for each training instance
        '''
        X_train_meta = self.generate_new_base_model_predictions(X, df, historical_df)
        return self.meta_model.predict(X_train_meta)
