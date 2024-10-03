import pandas as pd
import numpy as np
import config as cfg
from sota.mensa.model import MENSA
from utility.survival import (make_stratified_split, convert_to_structured,
                              make_time_bins, make_event_times, preprocess_data)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from tools.preprocessor import Preprocessor
from torch.utils.data import DataLoader, TensorDataset
import torch
import random
import warnings
from tools.data_loader import get_data_loader
from SurvivalEVAL.Evaluator import LifelinesEvaluator
from utility.evaluation import global_C_index, local_C_index
from utility.config import load_config
from SurvivalEVAL.Evaluations.util import KaplanMeier
from SurvivalEVAL import mean_error
from SurvivalEVAL.Evaluator import QuantileRegEvaluator

# Conformality
from tools.icp import IcpSurvival
from tools.scorer import SurvivalNC
from utility.conformal import OnsSideQuantileRegErrFunc

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Set precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = "proact"

def conformalize(
        nc,
        datasets,
        feature_names=[],
        n_events=1,
        condition=None,
        decensor_method="margin",
        n_quantiles=9,
        use_train=False
):
    trainset = datasets[0]
    valset = datasets[1]
    x_test = datasets[2]['X'].cpu().numpy()

    icp = IcpSurvival(nc, condition=condition,
                      decensor_method=decensor_method,
                      n_quantiles=n_quantiles)
    # Fit the ICP using the proper training set, and using valset for early stopping
    icp.fit(trainset, valset, feature_names)
    # Calibrate the ICP using the calibration set
    if use_train:
        valset = pd.concat([trainset, valset], ignore_index=True)
    
    event_quantiles, event_quan_preds = [], []
    for i in range(n_events): # for each event
        event_trainset = pd.DataFrame(trainset['X'], columns=feature_names)
        event_trainset['time'] = trainset['T'][:,i]
        event_trainset['event'] = trainset['E'][:,i]
        event_valset = pd.DataFrame(valset['X'], columns=feature_names)
        event_valset['time'] = valset['T'][:,i]
        event_valset['event'] = valset['E'][:,i]
        icp.calibrate(data_train=event_trainset, data_val=event_valset,
                      time_bins=time_bins, risk=i)
        quantiles, quan_preds = icp.predict(x_test, risk=i, time_bins=time_bins)
        event_quantiles.append(quantiles)
        event_quan_preds.append(quan_preds)

    return event_quantiles, event_quan_preds

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
        X_train, X_valid, X_test = preprocess_data(X_train, X_valid, X_test, cat_features,
                                                   num_features, as_array=False)
        feature_names = X_train.columns
        train_dict['X'] = torch.tensor(X_train.to_numpy(), device=device, dtype=dtype)
        train_dict['E'] = torch.tensor(train_dict['E'], device=device, dtype=torch.int64)
        train_dict['T'] = torch.tensor(train_dict['T'], device=device, dtype=torch.int64)
        valid_dict['X'] = torch.tensor(X_valid.to_numpy(), device=device, dtype=dtype)
        valid_dict['E'] = torch.tensor(valid_dict['E'], device=device, dtype=torch.int64)
        valid_dict['T'] = torch.tensor(valid_dict['T'], device=device, dtype=torch.int64)
        test_dict['X'] = torch.tensor(X_test.to_numpy(), device=device, dtype=dtype)
        test_dict['E'] = torch.tensor(test_dict['E'], device=device, dtype=torch.int64)
        test_dict['T'] = torch.tensor(test_dict['T'], device=device, dtype=torch.int64)
    
    n_samples = train_dict['X'].shape[0]
    n_features = train_dict['X'].shape[1]
    
    # Make time bins
    time_bins = make_time_bins(train_dict['T'].cpu(), event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))
    
    # Make model
    config = load_config(cfg.MENSA_CONFIGS_DIR, f"{dataset_name}.yaml")
    n_epochs = config['n_epochs']
    n_dists = config['n_dists']
    lr = config['lr']
    batch_size = config['batch_size']
    layers = config['layers']
    model = MENSA(n_features, layers=layers, n_events=n_events,
                  n_dists=n_dists, device=device)
    
    # Train NC
    error_func = OnsSideQuantileRegErrFunc()
    nc_model = SurvivalNC(model, error_func, config=config, device=device)
    datasets = [train_dict, valid_dict, test_dict]
    quan_levels, quan_preds = conformalize(nc_model, datasets, feature_names, n_events,
                                           decensor_method="sampling", condition=None) # sampling or margin
    
    # Make predictions
    all_preds = []
    for i in range(n_events):
        model_preds = model.predict(test_dict['X'].to(device), time_bins, risk=i)
        model_preds = pd.DataFrame(model_preds, columns=time_bins.cpu().numpy())
        all_preds.append(model_preds)

    # Make evaluation for each event
    for i, surv_pred in enumerate(all_preds):
        n_train_samples = len(train_dict['X'])
        n_test_samples= len(test_dict['X'])
        y_train_time = train_dict['T'][:,i].cpu().numpy()
        y_train_event = train_dict['E'][:,i].cpu().numpy()
        y_valid_time = valid_dict['T'][:,i].cpu().numpy()
        y_valid_event = valid_dict['E'][:,i].cpu().numpy()
        y_test_time = test_dict['T'][:,i].cpu().numpy()
        y_test_event = test_dict['E'][:,i].cpu().numpy()
        
        # Evaluate conformal prediction
        t_train_val = np.concatenate((y_train_time, y_valid_time))
        e_train_val = np.concatenate((y_train_event, y_valid_event))
        quantile_evaler = QuantileRegEvaluator(quan_preds[i], quan_levels[i],
                                               y_test_time, y_test_event,
                                               t_train_val, e_train_val,
                                               predict_time_method="Median", interpolation='Pchip')
        mae_margin = quantile_evaler.mae(method="Margin")
        ci = quantile_evaler.concordance()[0]
        ibs = quantile_evaler.integrated_brier_score(num_points=10)
        d_calib = quantile_evaler.d_calibration()[0]
        
        # Calculate KM estimate
        km_model = KaplanMeier(y_train_time, y_train_event)
        km_surv_prob = torch.from_numpy(km_model.predict(time_bins))
        time_idx = np.where(km_surv_prob <= 0.5, km_surv_prob, -np.inf).argmax(axis=0)
        km_estimate = np.array(len(y_test_time)*[float(time_bins[time_idx])])
        km_mae = mean_error(km_estimate, event_times=y_test_time, event_indicators=y_test_event,
                            train_event_times=y_train_time, train_event_indicators=y_train_event,
                            method='Margin')
        
        print(f"Evaluated E{i+1}: CI={round(ci, 3)}, IBS={round(ibs, 3)}, " +
              f"MAE={round(mae_margin, 3)}, D-Calib={round(d_calib, 3)}, " +
              f"KM MAE: {round(km_mae, 3)}")
