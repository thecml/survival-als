from SurvivalEVAL import mean_error
import pandas as pd
import numpy as np
import config as cfg
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
from sota_models import DeepSurv, train_deepsurv_model, make_deepsurv_prediction
from scipy.interpolate import interp1d
from SurvivalEVAL.Evaluator import LifelinesEvaluator
from utility.evaluation import global_C_index, local_C_index
from SurvivalEVAL.Evaluations.util import KaplanMeier

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

if __name__ == "__main__":
    # Load data
    dl = get_data_loader(dataset_name)
    dl = dl.load_data()
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
    
    # Train model
    config = dotdict(cfg.DEEPSURV_PARAMS)
    trained_models = []
    for i in range(n_events):
        model = DeepSurv(in_features=n_features, config=config)
        data_train = pd.DataFrame(train_dict['X'].cpu().numpy())
        data_train['time'] = train_dict['T'][:,i].cpu().numpy()
        data_train['event'] = train_dict['E'][:,i].cpu().numpy()
        data_valid = pd.DataFrame(valid_dict['X'].cpu().numpy())
        data_valid['time'] = valid_dict['T'][:,i].cpu().numpy()
        data_valid['event'] = valid_dict['E'][:,i].cpu().numpy()
        model = train_deepsurv_model(model, data_train, data_valid, time_bins, config=config,
                                     random_state=0, reset_model=True, device=device, dtype=dtype)
        trained_models.append(model)
    
    # Make predictions
    all_preds = []
    for trained_model in trained_models:
        preds, time_bins_model = make_deepsurv_prediction(trained_model, test_dict['X'].to(device),
                                                            config=config, dtype=dtype)
        spline = interp1d(time_bins_model.cpu().numpy(), preds.cpu().numpy(),
                            kind='linear', fill_value='extrapolate')
        preds = pd.DataFrame(spline(time_bins.cpu().numpy()), columns=time_bins.cpu().numpy())
        all_preds.append(preds)
    
    # Make evaluation for each event
    events = ['Speech', 'Salivation', 'Swallowing', "Handwriting", "Walking", 'Death']
    for i, surv_pred in enumerate(all_preds):
        n_train_samples = len(train_dict['X'])
        n_test_samples= len(test_dict['X'])
        y_train_time = train_dict['T'][:,i].cpu().numpy()
        y_train_event = train_dict['E'][:,i].cpu().numpy()
        y_test_time = test_dict['T'][:,i].cpu().numpy()
        y_test_event = test_dict['E'][:,i].cpu().numpy()
        
        lifelines_eval = LifelinesEvaluator(surv_pred.T, y_test_time, y_test_event,
                                            y_train_time, y_train_event)
        
        mae_margin = lifelines_eval.mae(method="Margin")
        ci = lifelines_eval.concordance()[0]
        ibs = lifelines_eval.integrated_brier_score()
        d_calib = lifelines_eval.d_calibration()[0]
        

        # Calculate KM estimate
        km_model = KaplanMeier(y_train_time, y_train_event)
        km_surv_prob = km_model.predict(time_bins)
        time_idx = np.where(km_surv_prob <= 0.5, km_surv_prob, -np.inf).argmax(axis=0)
        km_estimate = np.array(len(y_test_time)*[float(time_bins[time_idx])])
        km_mae = mean_error(km_estimate, event_times=y_test_time, event_indicators=y_test_event,
                            train_event_times=y_train_time, train_event_indicators=y_train_event,
                            method='Margin')
        
        print(f"Evaluated E{i+1}: CI={round(ci, 3)}, IBS={round(ibs, 3)}, " +
              f"MAE={round(mae_margin, 3)}, D-Calib={round(d_calib, 3)}, " +
              f"KM MAE: {round(km_mae, 3)}")
