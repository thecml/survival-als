from pathlib import Path
from SurvivalEVAL import mean_error
import pandas as pd
import numpy as np
import config as cfg
from utility.survival import (coverage, make_stratified_split, convert_to_structured,
                              make_time_bins, make_event_times, preprocess_data,
                              predict_median_survival_times)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from tools.preprocessor import Preprocessor
from torch.utils.data import DataLoader, TensorDataset
import torch
import random
import warnings
import math
from scipy.stats import chisquare
from tools.data_loader import get_data_loader
from SurvivalEVAL.Evaluator import LifelinesEvaluator
from SurvivalEVAL.Evaluations.util import KaplanMeier
from utility.evaluation import global_C_index, local_C_index
from utility.config import load_config

from bnn.model import BayesianMensa
from bnn.utility import train_model, make_ensemble_mensa_prediction

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = "proact"

if __name__ == "__main__":
    # Load data
    dl = get_data_loader(dataset_name)
    if dataset_name == "synthetic":
        dl = dl.load_data(cfg.SYNTHETIC_SETTINGS, device=device, dtype=dtype)
    else:
        dl = dl.load_data()
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1,
                                                      test_size=0.2, random_state=0)
    n_events = dl.n_events
    
    if dataset_name == "synthetic":
        for dataset in [train_dict, valid_dict, test_dict]: # put on device
            for key in ['X', 'T', 'E']:
                dataset[key] = dataset[key].to(device)
    else:
        # Preprocess data
        cat_features = dl.cat_features
        num_features = dl.num_features
        event_cols = [f'e{i+1}' for i in range(n_events)]
        time_cols = [f't{i+1}' for i in range(n_events)]
        X_train = pd.DataFrame(train_dict['X'], columns=dl.columns)
        X_valid = pd.DataFrame(valid_dict['X'], columns=dl.columns)
        X_test = pd.DataFrame(test_dict['X'], columns=dl.columns)
        X_train, X_valid, X_test= preprocess_data(X_train, X_valid, X_test, cat_features,
                                                  num_features, as_array=True)
        train_dict['X'] = torch.tensor(X_train, device=device, dtype=dtype)
        train_dict['E'] = torch.tensor(train_dict['E'], device=device, dtype=torch.int64)
        train_dict['T'] = torch.tensor(train_dict['T'], device=device, dtype=torch.int64)
        valid_dict['X'] = torch.tensor(X_valid, device=device, dtype=dtype)
        valid_dict['E'] = torch.tensor(valid_dict['E'], device=device, dtype=torch.int64)
        valid_dict['T'] = torch.tensor(valid_dict['T'], device=device, dtype=torch.int64)
        test_dict['X'] = torch.tensor(X_test, device=device, dtype=dtype)
        test_dict['E'] = torch.tensor(test_dict['E'], device=device, dtype=torch.int64)
        test_dict['T'] = torch.tensor(test_dict['T'], device=device, dtype=torch.int64)
    
    n_samples = train_dict['X'].shape[0]
    n_features = train_dict['X'].shape[1]
    
    # Make time bins
    time_bins = make_time_bins(train_dict['T'].cpu(), event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))
    
    # Training loop
    config = dotdict(load_config(cfg.BAYESIAN_CONFIGS_DIR, f"{dataset_name}.yaml"))
    n_dists = config['n_dists']
    layers = config['layers']
    model = BayesianMensa(n_features, n_dists, layers=layers,
                          n_events=n_events, config=config)
    model = train_model(model, train_dict, valid_dict, time_bins, config=config,
                        random_state=0, reset_model=True, device=device)
    
    # Make predictions
    X_test = test_dict['X']
    mean_preds, ensemble_preds = [], []
    for i in range(n_events):
        survival_outputs, time_bins, ensemble_outputs = make_ensemble_mensa_prediction(model, X_test, time_bins,
                                                                                       risk=i, n_dists=n_dists,
                                                                                       config=config)
        mean_model_preds = pd.DataFrame(survival_outputs, columns=time_bins.cpu().numpy()) # use mean survival outputs
        mean_preds.append(mean_model_preds)
        ensemble_preds.append(ensemble_outputs)
    
    # Make evaluation for each event
    for i, (mean_pred, ensemble_pred) in enumerate(zip(mean_preds, ensemble_preds)):
        y_train_time = train_dict['T'][:,i].cpu().numpy()
        y_train_event = train_dict['E'][:,i].cpu().numpy()
        y_test_time = test_dict['T'][:,i].cpu().numpy()
        y_test_event = test_dict['E'][:,i].cpu().numpy()
        
        lifelines_eval = LifelinesEvaluator(mean_pred.T, y_test_time, y_test_event,
                                            y_train_time, y_train_event)
        
        mae_margin = lifelines_eval.mae(method="Margin")
        ci = lifelines_eval.concordance()[0]
        ibs = lifelines_eval.integrated_brier_score(num_points=len(time_bins))
        d_calib = lifelines_eval.d_calibration()[0]
        
        # Calculate KM estimate
        km_model = KaplanMeier(y_train_time, y_train_event)
        km_surv_prob = torch.from_numpy(km_model.predict(time_bins))
        time_idx = np.where(km_surv_prob <= 0.5, km_surv_prob, -np.inf).argmax(axis=0)
        km_estimate = np.array(len(y_test_time)*[float(time_bins[time_idx])])
        km_mae = mean_error(km_estimate, event_times=y_test_time, event_indicators=y_test_event,
                            train_event_times=y_train_time, train_event_indicators=y_train_event,
                            method='Margin')
        
        # Calculate C-calib
        coverage_stats = {}
        credible_region_sizes = np.arange(0.1, 1, 0.1)
        for percentage in credible_region_sizes:
            drop_num = math.floor(0.5 * config.n_samples_test * (1 - percentage))
            lower_outputs = torch.kthvalue(ensemble_pred, k=1 + drop_num, dim=0)[0]
            upper_outputs = torch.kthvalue(ensemble_pred, k=config.n_samples_test - drop_num, dim=0)[0]
            coverage_stats[percentage] = coverage(time_bins, upper_outputs, lower_outputs,
                                                  y_test_time, y_test_event)
        expected_percentages = coverage_stats.keys()
        observed_percentages = coverage_stats.values()
        expected = [x / sum(expected_percentages) * 100 for x in expected_percentages] # normalize
        observed = [x / sum(observed_percentages) * 100 for x in observed_percentages]
        _, p_value = chisquare(f_obs=observed, f_exp=expected)
        c_calib = p_value
        
        print(f"Evaluated E{i+1}: CI={round(ci, 3)}, IBS={round(ibs, 3)}, " +
              f"MAE={round(mae_margin, 3)}, D-Calib={round(d_calib, 3)}, " +
              f"C-Calib={round(c_calib, 3)}, KM MAE: {round(km_mae, 3)}")
        