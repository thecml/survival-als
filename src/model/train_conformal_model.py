from pathlib import Path
import pandas as pd
import numpy as np
import config as cfg
from conformal.model import ConformalMensa
from utility.survival import make_time_bins, preprocess_data
import torch
import random
from tools.data_loader import get_data_loader
from utility.config import load_config
from SurvivalEVAL.Evaluations.util import KaplanMeier
from SurvivalEVAL import mean_error
from SurvivalEVAL.Evaluator import QuantileRegEvaluator
import joblib

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Set precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = "proact"

if __name__ == "__main__":
    # Load data
    dl = get_data_loader(dataset_name).load_data()
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1,
                                                      test_size=0.2, random_state=0)
    n_events = dl.n_events
    
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
    
    # Train model
    config = load_config(cfg.MENSA_CONFIGS_DIR, f"{dataset_name}.yaml")
    model = ConformalMensa(n_features, time_bins=time_bins,
                           n_events=n_events, device=device, config=config)
    datasets = [train_dict, valid_dict]
    model.fit_calibrate(datasets, feature_names, decensor_method="margin", condition=None)
    
    # Save model
    path = Path.joinpath(cfg.MODELS_DIR, f"conformal_{dataset_name}.pkl")
    joblib.dump(model, path)
    