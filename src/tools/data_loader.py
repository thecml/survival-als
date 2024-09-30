import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
import config as cfg
import numpy as np
from utility.dgp import DGP_Weibull_linear, DGP_Weibull_nonlinear
from utility.survival import make_stratified_split
import torch
import random

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
    Data loader for ALS dataset (ME). Use the PRO-ACT dataset.
    """
    def load_data(self, n_samples:int = None):
        df = pd.read_csv(f'{cfg.PROACT_DATA_DIR}/proact_processed.csv', index_col=0)
        if n_samples:
            df = df.sample(n=n_samples, random_state=0)
        columns_to_drop = [col for col in df.columns if
                           any(substring in col for substring in ['Event', 'TTE'])]
        df = df.loc[(df['TTE_Speech'] > 0) & (df['TTE_Swallowing'] > 0)
                    & (df['TTE_Handwriting'] > 0) & (df['TTE_Walking'] > 0)
                    & (df['TTE_Death'] > 0)]
        df = df.loc[(df['TTE_Speech'] <= 1800) & (df['TTE_Swallowing'] <= 1800)
                    & (df['TTE_Handwriting'] <= 1800) & (df['TTE_Walking'] <= 1800)
                    & (df['TTE_Death'] <= 1800)] # 5 years max
        df = df.drop(df.filter(like='_Strength').columns, axis=1) # drop strength cols as they have many nans
        df['El_escorial'] = df['El_escorial'].replace('Possible', 'Probable') # Replace "Possible" with "Probable"
        events = ['Speech', 'Swallowing', 'Handwriting', 'Walking', 'Death']
        self.X = df.drop(columns_to_drop, axis=1)
        self.columns = list(self.X.columns)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        times = [df[f'TTE_{event_col}'].values for event_col in events]
        events = [df[f'Event_{event_col}'].values for event_col in events]
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
        df['time'] = self.y_t[:,0] # split on first time
        
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='time', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            random_state=random_state)
        
        dataframes = [df_train, df_valid, df_test]
        event_cols = ['e1', 'e2', 'e3', 'e4', 'e5']
        time_cols = ['t1', 't2', 't3', 't4', 't5']
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = dataframe.drop(event_cols + time_cols + ['time'], axis=1).values
            data_dict['E'] = np.stack([dataframe['e1'].values, dataframe['e2'].values,
                                       dataframe['e3'].values, dataframe['e4'].values,
                                       dataframe['e5'].values], axis=1).astype(np.int64)
            data_dict['T'] = np.stack([dataframe['t1'].values, dataframe['t2'].values,
                                       dataframe['t3'].values, dataframe['t4'].values,
                                       dataframe['t5'].values], axis=1).astype(np.int64)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]
    
class CALSNICDataLoader(BaseDataLoader):
    def load_data(self, n_samples:int = None):
        df = pd.read_csv(f'{cfg.CALSNIC_DATA_DIR}/calsnic_processed.csv', index_col=0)
        if n_samples:
            df = df.sample(n=n_samples, random_state=0)
        columns_to_drop = [col for col in df.columns if
                           any(substring in col for substring in ['Event', 'TTE'])]
        events = ['Speech', 'Swallowing', 'Handwriting', 'Walking', 'Death']
        self.X = df[['Visit', 'Symptom_Duration', 'CNSLS_TotalScore', 'TAP_Fingertapping_Right_avg',
                     'TAP_Fingertapping_Left_avg', 'TAP_Foottapping_Right_avg', 'Region_of_Onset',
                     'TAP_Foottapping_Left_avg', 'UMN_Right', 'UMN_Left', 'Age', 'SymptomDays']]
        self.columns = list(self.X.columns)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        times = [df[f'TTE_{event_col}'].values for event_col in events]
        events = [df[f'Event_{event_col}'].values for event_col in events]
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
        df['time'] = self.y_t[:,0] # split on first time
        
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='time', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            random_state=random_state)
        
        dataframes = [df_train, df_valid, df_test]
        event_cols = ['e1', 'e2', 'e3', 'e4', 'e5']
        time_cols = ['t1', 't2', 't3', 't4', 't5']
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = dataframe.drop(event_cols + time_cols + ['time'], axis=1).values
            data_dict['E'] = np.stack([dataframe['e1'].values, dataframe['e2'].values,
                                       dataframe['e3'].values, dataframe['e4'].values,
                                       dataframe['e5'].values], axis=1).astype(np.int64)
            data_dict['T'] = np.stack([dataframe['t1'].values, dataframe['t2'].values,
                                       dataframe['t3'].values, dataframe['t4'].values,
                                       dataframe['t5'].values], axis=1).astype(np.int64)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]
    
class SyntheticDataLoader(BaseDataLoader):
    def load_data(self, data_config, linear=True, device='cpu', dtype=torch.float64):
        """
        This method generates synthetic data for 3 multiple events (with adm. censoring)
        DGP1: Data generation process for event 1
        DGP2: Data generation process for event 2
        DGP3: Data generation process for event 3
        """
        alpha_e1 = data_config['alpha_e1']
        alpha_e2 = data_config['alpha_e2']
        alpha_e3 = data_config['alpha_e3']
        gamma_e1 = data_config['gamma_e1']
        gamma_e2 = data_config['gamma_e2']
        gamma_e3 = data_config['gamma_e3']
        n_samples = data_config['n_samples']
        n_features = data_config['n_features']
        adm_censoring_time = data_config['adm_censoring_time']
        
        X = torch.rand((n_samples, n_features), device=device, dtype=dtype)
        
        if linear:
            dgp1 = DGP_Weibull_linear(n_features, alpha_e1, gamma_e1, device, dtype)
            dgp2 = DGP_Weibull_linear(n_features, alpha_e2, gamma_e2, device, dtype)
            dgp3 = DGP_Weibull_linear(n_features, alpha_e3, gamma_e3, device, dtype)
        else:
            dgp1 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e1, gamma=gamma_e1, device=device, dtype=dtype)
            dgp2 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e2, gamma=gamma_e2, device=device, dtype=dtype)
            dgp3 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e3, gamma=gamma_e3, device=device, dtype=dtype)

        rng = np.random.default_rng(0)
        u = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
        v = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
        w = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
        uvw = torch.stack([u, v, w], dim=1)

        t1_times = dgp1.rvs(X, uvw[:,0]).cpu()
        t2_times = dgp2.rvs(X, uvw[:,1]).cpu()
        t3_times = dgp3.rvs(X, uvw[:,2]).cpu()
        
        # Make adm. censoring
        event_times = np.stack([t1_times, t2_times, t3_times], axis=1)
        event_times = np.minimum(event_times, adm_censoring_time)
        event_indicators = (event_times < adm_censoring_time).astype(int)

        # Format data
        columns = [f'X{i}' for i in range(n_features)]
        self.X = pd.DataFrame(X.cpu(), columns=columns)
        self.y_t = event_times
        self.y_e = event_indicators
        self.dgps = [dgp1, dgp2, dgp3]
        self.n_events = 3
        
        return self

    def split_data(self, train_size: float, valid_size: float,
                   test_size: float, dtype=torch.float64, random_state=0):
        df = pd.DataFrame(self.X)
        df['e1'] = self.y_e[:,0]
        df['e2'] = self.y_e[:,1]
        df['e3'] = self.y_e[:,2]
        df['t1'] = self.y_t[:,0]
        df['t2'] = self.y_t[:,1]
        df['t3'] = self.y_t[:,2]
        df['time'] = self.y_t[:,0] # split on first time
        
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='time', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            random_state=random_state)
        
        dataframes = [df_train, df_valid, df_test]
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = torch.tensor(dataframe.loc[:, 'X0':'X9'].values, dtype=dtype)
            data_dict['E'] = torch.stack([torch.tensor(dataframe['e1'].values, dtype=dtype),
                                          torch.tensor(dataframe['e2'].values, dtype=dtype),
                                          torch.tensor(dataframe['e3'].values, dtype=dtype)], axis=1)
            data_dict['T'] = torch.stack([torch.tensor(dataframe['t1'].values, dtype=dtype),
                                          torch.tensor(dataframe['t2'].values, dtype=dtype),
                                          torch.tensor(dataframe['t3'].values, dtype=dtype)], axis=1)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]
    
def get_data_loader(dataset_name:str) -> BaseDataLoader:
    if dataset_name == "proact":
        return PROACTDataLoader()
    elif dataset_name == "calsnic":
        return CALSNICDataLoader()
    elif dataset_name == "synthetic":
        return SyntheticDataLoader()
    else:
        raise ValueError("Dataset not found")