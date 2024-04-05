import numpy as np
import pandas as pd
from sksurv.datasets import load_veterans_lung_cancer, load_gbsg2, load_aids, load_whas500, load_flchain
from sklearn.model_selection import train_test_split
#import shap
from abc import ABC, abstractmethod
from typing import Tuple, List
#from tools.preprocessor import Preprocessor
from pathlib import Path
from utility.survival import convert_to_structured
import config as cfg

class DataLoader():
    """
    Data loader
    """
    def __init__(self):
        """Initilizer method that takes a file path, file name,
        settings and optionally a converter"""
        self.X: pd.DataFrame = None
        self.y: np.ndarray = None
        self.num_features: List[str] = None
        self.cat_features: List[str] = None

    def get_data(self) -> pd.DataFrame:
        """
        This method returns the features and targets
        :return: df
        """
        df = pd.DataFrame(self.X)
        df['time'] = self.y['time']
        df['event'] = self.y['event']
        return df

    def get_features(self) -> List[str]:
        """
        This method returns the names of numerical and categorial features
        :return: the columns of X as a list
        """
        return self.num_features, self.cat_features

    def _get_num_features(self, data) -> List[str]:
        return data.select_dtypes(include=np.number).columns.tolist()

    def _get_cat_features(self, data) -> List[str]:
        return data.select_dtypes(['category']).columns.tolist()

    def load_data(self, event: str):
        """
        This method loads and prepares the data for modeling
        event = {Speech, Swallowing, Handwriting, Walking}
        """
        if event == "Speech":
            df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, f"data_speech.csv"), index_col=0)
            y_label = "Event_ALSFRS_1_Speech"
        elif event == "Swallowing":
            df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, f"data_swallowing.csv"), index_col=0)
            y_label = "Event_ALSFRS_3_Swallowing"
        elif event == "Handwriting":
            df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, f"data_handwriting.csv"), index_col=0)
            y_label = "Event_ALSFRS_4_Handwriting"
        elif event == "Walking":
            df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, f"data_walking.csv"), index_col=0)
            y_label = "Event_ALSFRS_8_Walking"
        else:
            raise ValueError("Invalid event, please select {Speech, Swallowing, Handwriting, Walking}")
        
        X = df[['SymptomDays', 'Region_of_Onset', 'DiseaseProgressionRate',
                'TAP_Fingertapping_Right_avg', 'TAP_Foottapping_Right_avg',
                'UMN_Right', 'UMN_Left']].copy(deep=True)
        
        obj_cols = X.select_dtypes(['bool']).columns.tolist() \
                   + X.select_dtypes(['object']).columns.tolist()
        for col in obj_cols:
            X[col] = X[col].astype('category')
        self.X = pd.DataFrame(X)

        self.y = convert_to_structured(df['TTE'], df[y_label])
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        
        return self