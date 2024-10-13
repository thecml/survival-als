import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

from tqdm import trange

from sota.mensa.loss import conditional_weibull_loss, conditional_weibull_loss_multi
from sota.mensa.model import MENSA
from tools.icp import IcpSurvival
from tools.scorer import SurvivalNC
from utility.conformal import OnsSideQuantileRegErrFunc

class ConformalMensa:
    """
    Wrapper that implements a conformal version of MENSA
    n_features: number of features
    n_events: number of events (K)
    n_dists: number of Weibull distributions
    layers: layers and size of the network, e.g., [32, 32].
    device: device to use, e.g., cpu or cuda
    """
    def __init__(self, n_features, time_bins, n_events, device, config):
        self.n_features = n_features
        self.n_events = n_events
        self.device = device
        
        self.time_bins = time_bins
        
        n_dists = config['n_dists']
        layers = config['layers']
        
        error_func = OnsSideQuantileRegErrFunc()
        self.mensa_model = MENSA(n_features, layers=layers,
                                 n_events=n_events,
                                 n_dists=n_dists, device=device)
        self.nc_model = SurvivalNC(self.mensa_model, error_func, config=config, device=device)
        
    def fit_calibrate(self, datasets, feature_names=[], condition=None,
                      decensor_method="margin", n_quantiles=9, use_train=False, verbose=False):
        trainset = datasets[0]
        valset = datasets[1]

        # Fit the ICP using the proper training set, and using valid set for early stopping
        self.icp = IcpSurvival(self.nc_model, condition=condition,
                               decensor_method=decensor_method,
                               n_quantiles=n_quantiles)
        self.icp.fit(trainset, valset, feature_names, verbose)
        
        # Calibrate the ICP using the calibration set
        if use_train:
            valset = pd.concat([trainset, valset], ignore_index=True)
        
        for i in range(self.n_events): # for each event
            event_trainset = pd.DataFrame(trainset['X'], columns=feature_names)
            event_trainset['time'] = trainset['T'][:,i]
            event_trainset['event'] = trainset['E'][:,i]
            event_valset = pd.DataFrame(valset['X'], columns=feature_names)
            event_valset['time'] = valset['T'][:,i]
            event_valset['event'] = valset['E'][:,i]
            self.icp.calibrate(data_train=event_trainset, data_val=event_valset,
                               time_bins=self.time_bins, risk=i)

    def predict(self, X_test: dict):
        x_test = X_test['X'].cpu().numpy()
        event_quantiles, event_quan_preds = [], []
        for i in range(self.n_events):
            quantiles, quan_preds = self.icp.predict(x_test, risk=i, time_bins=self.time_bins)
            event_quantiles.append(quantiles)
            event_quan_preds.append(quan_preds)
        return event_quantiles, event_quan_preds