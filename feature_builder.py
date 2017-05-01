import pandas as pd
import numpy as np

class FeatureBuilder:

    def __init__(self, non_historical_features, historical_features):
        '''Initializes a feature building instance that stores all feature function to create features using historic and non historic features.
        -----------

        non_historical_features: list
        list of feature functions that build new features by manipulating data without knowledge that exists at all time

        historical_features: list
        list of feature functions that build new features by manipulating data without knowledge that only exists in the past

        Returns:
        --------
        -
        '''
        self.non_historical_features = non_historical_features
        self.historical_features = historical_features

    def create_historical_features(self, df, historical_df):
        '''Create all historical features by iterating over the historical feature functions with which the object was instantiated.
        -----------

        df: pd.DataFrame
        The dataframe that contains all data.

        historical_df: pd.DataFrame
        The dataframe that contains only historic data.

        Returns:
        --------
        np.array
        All historic Features in a numpy array of dim n x m where n is the amount of rows in df and m the amount of created features by
        all historic feature functions.
        '''
        historical_features = [function(df,historical=historical_df) for function in self.historical_features]
        return np.hstack([x.reshape(len(x),1) for x in historical_features])

    def create_non_historical_features(self, df, save_features=False, save_path=''):
        '''Create all historical features by iterating over the historical feature functions with which the object was instantiated.
        -----------

        df: pd.DataFrame
        The dataframe that contains all data.

        save_features: Boolean
        Indicates if the features should be stored on disk as .npy file.

        save_path: string
        Only necessary if features should be saved and therefore represents filepath to the file to be stored.

        Returns:
        --------
        np.array
        All non historic features in a numpy array of dim n x m where n is the amount of rows in df and m the amount of created features by
        all non historic feature functions.
        '''
        non_historical_features = [function(df) for function in self.non_historical_features]
        if save_features:
            np.save(save_path, np.hstack(non_historical_features))
        else:
            return np.hstack(non_historical_features)
