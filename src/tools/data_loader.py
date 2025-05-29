import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List
import config as cfg
import numpy as np
from utility.survival import make_stratified_split
import torch

class BaseDataLoader(ABC):
    """
    Base class for data loaders.
    """
    def __init__(self):
        """Initilizer method that takes a file path, file name,
        settings and optionally a converter"""
        self.X: pd.DataFrame = None
        self.y_t: List[np.ndarray] = None
        self.y_e: List[np.ndarray] = None
        self.num_features: List[str] = None
        self.cat_features: List[str] = None
        self.min_time = None
        self.max_time = None
        self.n_events = None
        self.params = None

    @abstractmethod
    def load_data(self, n_samples) -> None:
        """Loads the data from a data set at startup"""
        
    @abstractmethod
    def split_data(self) -> None:
        """Loads the data from a data set at startup"""

    def get_data(self) -> pd.DataFrame:
        """
        This method returns the features and targets
        :returns: X, y_t and y_e
        """
        return self.X, self.y_t, self.y_e

    def get_features(self) -> List[str]:
        """
        This method returns the names of numerical and categorial features
        :return: the columns of X as a list
        """
        return self.num_features, self.cat_features

    def _get_num_features(self, data) -> List[str]:
        return data.select_dtypes(include=np.number).columns.tolist()

    def _get_cat_features(self, data) -> List[str]:
        return data.select_dtypes(['object']).columns.tolist()
    
class PROACTDataLoader(BaseDataLoader):
    """
    Data loader for ALS dataset (ME). Uses the PRO-ACT dataset.
    """
    def load_data(self, n_samples:int = None):
        df = pd.read_csv(f'{cfg.PROACT_DATA_DIR}/proact_processed.csv', index_col=0)
        if n_samples:
            df = df.sample(n=n_samples, random_state=0)
        label_cols = [col for col in df.columns if any(substring in col for substring in ['Event', 'TTE'])]
        event_names = ['Speech', 'Swallowing', 'Handwriting', 'Walking', 'Dyspnea']
        for event_name in event_names:
            df = df.loc[(df[f'Event_{event_name}'] == 0) | (df[f'Event_{event_name}'] == 1)] # drop already occured
            df = df.loc[(df[f'TTE_{event_name}'] > 0) & (df[f'TTE_{event_name}'] <= 500)] # 1 - 500
        df = df.drop(df.filter(like='_Strength').columns, axis=1) # Drop strength tests
        df = df.drop('Race_Caucasian', axis=1) # Drop race information
        df = df.drop('El_escorial', axis=1) # Drop el_escorial
        df = df.drop(['Height', 'Weight', 'BMI'], axis=1) # Drop height/weight/bmi
        self.X = df.drop(label_cols, axis=1)
        self.columns = list(self.X.columns)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        times = [df[f'TTE_{event_col}'].values for event_col in event_names]
        events = [df[f'Event_{event_col}'].values for event_col in event_names]
        self.y_t = np.stack((times[0], times[1], times[2], times[3], times[4]), axis=1)
        self.y_e = np.stack((events[0], events[1], events[2], events[3], events[4]), axis=1)
        self.n_events = 5
        return self

    def split_data(self, train_size: float, valid_size: float,
                   test_size: float, dtype=torch.float64, random_state=0):
        df = pd.DataFrame(self.X)
        df['e1'] = self.y_e[:,0]
        df['e2'] = self.y_e[:,1]
        df['e3'] = self.y_e[:,2]
        df['e4'] = self.y_e[:,3]
        df['e5'] = self.y_e[:,4]
        df['t1'] = self.y_t[:,0]
        df['t2'] = self.y_t[:,1]
        df['t3'] = self.y_t[:,2]
        df['t4'] = self.y_t[:,3]
        df['t5'] = self.y_t[:,4]
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='multi', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            random_state=random_state, n_events=4)
        
        dataframes = [df_train, df_valid, df_test]
        event_cols = ['e1', 'e2', 'e3', 'e4', 'e5']
        time_cols = ['t1', 't2', 't3', 't4', 't5']
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = dataframe.drop(event_cols + time_cols, axis=1).values
            data_dict['E'] = np.stack([dataframe['e1'].values, dataframe['e2'].values,
                                       dataframe['e3'].values, dataframe['e4'].values,
                                       dataframe['e5'].values], axis=1).astype(np.int64)
            data_dict['T'] = np.stack([dataframe['t1'].values, dataframe['t2'].values,
                                       dataframe['t3'].values, dataframe['t4'].values,
                                       dataframe['t5'].values], axis=1).astype(np.int64)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]
    
def get_data_loader(dataset_name:str) -> BaseDataLoader:
    if dataset_name == "proact":
        return PROACTDataLoader()
    else:
        raise ValueError("Dataset not found")