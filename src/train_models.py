import argparse
import os
from SurvivalEVAL import mean_error
import pandas as pd
import numpy as np
import config as cfg
from conformal.model import ConformalMensa
from sota.deephit.utility import format_data_deephit_single, make_deephit_single, train_deephit_model
from sota.deepsurv.model import DeepSurv
from sota.deepsurv.utility import make_deepsurv_prediction, train_deepsurv_model
from sota.hierarchical.helper import format_hierarchical_hyperparams
from sota.mtlr.model import mtlr
from sota.mtlr.utility import make_mtlr_prediction, train_mtlr_model
from utility.data import calculate_layer_size_hierarch, format_hierarchical_data_me
from utility.survival import (coverage, make_stratified_split, convert_to_structured,
                              make_time_bins, make_event_times, preprocess_data,
                              predict_median_survival_times)
import torch
import random
from scipy.interpolate import interp1d
from tools.data_loader import get_data_loader
from SurvivalEVAL.Evaluator import LifelinesEvaluator
from SurvivalEVAL.Evaluations.util import KaplanMeier
from utility.config import load_config
from sota.hierarchical import util
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

from sota.mensa.model import MENSA
from utility.conformal import quantile_to_survival

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

# Define models
MODELS = ['coxph', 'rsf', 'deepsurv', 'mtlr', 'mensa']
          
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_name', type=str, default='proact')
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    seed = args.seed
    dataset_name = args.dataset_name

    # Load and split data
    dl = get_data_loader(dataset_name)
    dl = dl.load_data()
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1,
                                                      test_size=0.2, random_state=seed)
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
    
    # Evaluate models
    for model_name in MODELS:
        if model_name == "coxph":
            config = dotdict(cfg.COXPH_PARAMS)
            alpha = config['alpha']
            n_iter = config['n_iter']
            tol = config['tol']
            trained_models = []
            for i in range(n_events):
                X_train = pd.DataFrame(train_dict['X'].cpu().numpy(), columns=feature_names)
                y_train = convert_to_structured(train_dict['T'][:,i].cpu().numpy(), train_dict['E'][:,i].cpu().numpy())
                model = CoxPHSurvivalAnalysis(alpha=alpha, n_iter=n_iter, tol=tol)
                model.fit(X_train, y_train)
                trained_models.append(model)
        elif model_name == "rsf":
            config = dotdict(cfg.RSF_PARAMS)
            n_estimators = config['n_estimators']
            max_depth = config['max_depth']
            min_samples_split = config['min_samples_split']
            min_samples_leaf =  config['min_samples_leaf']
            max_features = config['max_features']
            trained_models = []
            for i in range(n_events):
                X_train = pd.DataFrame(train_dict['X'].cpu().numpy(), columns=feature_names)
                y_train = convert_to_structured(train_dict['T'][:,i].cpu().numpy(), train_dict['E'][:,i].cpu().numpy())
                model = RandomSurvivalForest(random_state=0, n_estimators=n_estimators, max_depth=max_depth,
                                             min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                             max_features=max_features)
                model.fit(X_train, y_train)
                trained_models.append(model)
        elif model_name == "deepsurv":
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
        elif model_name == "deephit":
            config = dotdict(cfg.DEEPHIT_PARAMS)
            trained_models = []
            for i in range(n_events):
                model = make_deephit_single(in_features=n_features, out_features=len(time_bins),
                                            time_bins=time_bins.cpu().numpy(), device=device, config=config)
                labtrans = model.label_transform
                train_data, valid_data, out_features, duration_index = format_data_deephit_single(train_dict, valid_dict, labtrans, event_id=i)
                model = train_deephit_model(model, train_data['X'], (train_data['T'], train_data['E']),
                                            (valid_data['X'], (valid_data['T'], valid_data['E'])), config)
                trained_models.append(model)
        elif model_name == "hierarch":
            config = load_config(cfg.HIERARCH_CONFIGS_DIR, f"{dataset_name}.yaml")
            n_time_bins = len(time_bins)
            train_data, valid_data, test_data = format_hierarchical_data_me(train_dict, valid_dict, test_dict, n_time_bins)
            config['min_time'] = int(train_data[1].min())
            config['max_time'] = int(train_data[1].max())
            config['num_bins'] = n_time_bins
            params = cfg.HIERARCH_PARAMS
            params['n_batches'] = int(n_samples/params['batch_size'])
            layer_size = params['layer_size_fine_bins'][0][0]
            params['layer_size_fine_bins'] = calculate_layer_size_hierarch(layer_size, n_time_bins)
            hyperparams = format_hierarchical_hyperparams(params)
            verbose = params['verbose']
            model = util.get_model_and_output("hierarch_full", train_data, test_data, valid_data, config, hyperparams, verbose)
        elif model_name == "mtlr":
            config = dotdict(cfg.MTLR_PARAMS)
            trained_models = []
            for i in range(n_events):
                X_train = pd.DataFrame(train_dict['X'].cpu().numpy(), columns=feature_names)
                X_valid = pd.DataFrame(valid_dict['X'].cpu().numpy(), columns=feature_names)
                y_train = convert_to_structured(train_dict['T'][:,i].cpu().numpy(), train_dict['E'][:,i].cpu().numpy())
                y_valid = convert_to_structured(valid_dict['T'][:,i].cpu().numpy(), valid_dict['E'][:,i].cpu().numpy())
                data_train = X_train.copy()
                data_train["time"] = pd.Series(y_train['time'])
                data_train["event"] = pd.Series(y_train['event']).astype(int)
                data_valid = X_valid.copy()
                data_valid["time"] = pd.Series(y_valid['time'])
                data_valid["event"] = pd.Series(y_valid['event']).astype(int)
                config = dotdict(cfg.MTLR_PARAMS)
                num_time_bins = len(time_bins)
                model = mtlr(in_features=n_features, num_time_bins=num_time_bins, config=config)
                model = train_mtlr_model(model, data_train, data_valid, time_bins.cpu().numpy(),
                                         config, random_state=0, dtype=dtype,
                                         reset_model=True, device=device)
                trained_models.append(model)
        elif model_name == "mensa":
            config = dotdict(load_config(cfg.MENSA_CONFIGS_DIR, f"{dataset_name}.yaml"))
            n_epochs = config['n_epochs']
            n_dists = config['n_dists']
            lr = config['lr']
            batch_size = config['batch_size']
            layers = config['layers']
            model = MENSA(n_features, n_events, n_dists=n_dists, layers=layers, device=device)
            model.fit(train_dict, valid_dict, learning_rate=lr, n_epochs=n_epochs,
                      patience=10, batch_size=batch_size, verbose=False)
        elif model_name == "conformal":
            config = load_config(cfg.MENSA_CONFIGS_DIR, f"{dataset_name}.yaml")
            model = ConformalMensa(n_features, time_bins=time_bins, n_events=n_events, device=device, config=config)
            datasets = [train_dict, valid_dict]
            feature_names = X_train.columns
            model.fit_calibrate(datasets, feature_names, decensor_method="margin", condition=None, verbose=False)
        else:
            raise NotImplementedError()
    
        # Compute survival function
        if model_name == "coxph":
            X_test = pd.DataFrame(test_dict['X'].cpu().numpy(), columns=feature_names)
            all_preds = []
            for trained_model in trained_models:
                model_preds = trained_model.predict_survival_function(X_test)
                model_preds = np.row_stack([fn(trained_model.unique_times_) for fn in model_preds])
                spline = interp1d(trained_model.unique_times_, model_preds,
                                  kind='linear', fill_value='extrapolate')
                preds = pd.DataFrame(spline(time_bins.cpu().numpy()), columns=time_bins.cpu().numpy())
                preds.iloc[:, 0] = 1 # ensure t=0 is 1
                all_preds.append(preds)
        elif model_name == "rsf":
            X_test = pd.DataFrame(test_dict['X'].cpu().numpy(), columns=feature_names)
            all_preds = []
            for trained_model in trained_models:
                model_preds = trained_model.predict_survival_function(X_test)
                model_preds = np.row_stack([fn(trained_model.unique_times_) for fn in model_preds])
                spline = interp1d(trained_model.unique_times_, model_preds,
                                  kind='linear', fill_value='extrapolate')
                preds = pd.DataFrame(spline(time_bins.cpu().numpy()), columns=time_bins.cpu().numpy())
                preds.iloc[:, 0] = 1 # ensure t=0 is 1
                all_preds.append(preds)
        elif model_name == "deepsurv":
            all_preds = []
            for trained_model in trained_models:
                preds, time_bins_model = make_deepsurv_prediction(trained_model, test_dict['X'].to(device),
                                                                  config=config, dtype=dtype)
                spline = interp1d(time_bins_model.cpu().numpy(), preds.cpu().numpy(),
                                  kind='linear', fill_value='extrapolate')
                preds = pd.DataFrame(spline(time_bins.cpu().numpy()), columns=time_bins.cpu().numpy())
                all_preds.append(preds)
        elif model_name == "deephit":
            all_preds = []
            for trained_model in trained_models:
                preds = trained_model.predict_surv(test_dict['X']).cpu().numpy()
                preds = pd.DataFrame(preds, columns=time_bins.cpu().numpy())
                all_preds.append(preds)
        elif model_name == "hierarch":
            event_preds = util.get_surv_curves(torch.tensor(test_data[0], dtype=dtype), model)
            bin_locations = np.linspace(0, config['max_time'], event_preds[0].shape[1])
            all_preds = []
            for i in range(len(event_preds)):
                preds = pd.DataFrame(event_preds[i], columns=bin_locations)
                all_preds.append(preds)
        elif model_name == "mtlr":
            all_preds = []
            for i, trained_model in enumerate(trained_models):
                X_test = pd.DataFrame(test_dict['X'].cpu().numpy(), columns=feature_names)
                y_test = convert_to_structured(test_dict['T'][:,i].cpu().numpy(), test_dict['E'][:,i].cpu().numpy())
                data_test = X_test.copy()
                data_test["time"] = pd.Series(y_test['time'])
                data_test["event"] = pd.Series(y_test['event']).astype('int')
                mtlr_test_data = torch.tensor(data_test.drop(["time", "event"], axis=1).values,
                                              dtype=dtype, device=device)
                survival_outputs, _, _ = make_mtlr_prediction(trained_model, mtlr_test_data, time_bins, config)
                preds = survival_outputs[:, 1:].cpu().numpy()
                preds = pd.DataFrame(preds, columns=time_bins.cpu().numpy())
                all_preds.append(preds)
        elif model_name == "mensa":
            all_preds = []
            for i in range(n_events):
                preds = model.predict(test_dict['X'].to(device), time_bins, risk=i)
                preds = pd.DataFrame(preds, columns=time_bins.cpu().numpy())
                all_preds.append(preds)   
        elif model_name == "conformal":
            all_preds = []
            quan_levels, quan_preds = model.predict(test_dict['X'])
            for i in range(n_events):
                preds = quantile_to_survival(quan_levels[i], quan_preds[i], time_bins)
                preds = pd.DataFrame(preds, columns=time_bins.cpu().numpy())
                all_preds.append(preds)
        else:
            raise NotImplementedError()
    
        # Evaluate predictions each event
        result_cols = ["DatasetName", "ModelName", "Seed", "EventId",
                       "CI", "IBS", "MAEM", "MAEKM", "DCalib"]
        for event_id, surv_pred in enumerate(all_preds):
            y_train_time = train_dict['T'][:,event_id].cpu().numpy()
            y_train_event = train_dict['E'][:,event_id].cpu().numpy()
            y_test_time = test_dict['T'][:,event_id].cpu().numpy()
            y_test_event = test_dict['E'][:,event_id].cpu().numpy()
            lifelines_eval = LifelinesEvaluator(surv_pred.T, y_test_time, y_test_event,
                                                y_train_time, y_train_event)
        
            mae_margin = lifelines_eval.mae(method="Margin")
            ci = lifelines_eval.concordance()[0]
            ibs = lifelines_eval.integrated_brier_score(num_points=10)
            d_calib = lifelines_eval.d_calibration()[0]
            
            # Calculate KM estimate
            km_model = KaplanMeier(y_train_time, y_train_event)
            km_surv_prob = torch.from_numpy(km_model.predict(time_bins.cpu().numpy()))
            time_idx = np.where(km_surv_prob <= 0.5, km_surv_prob, -np.inf).argmax(axis=0)
            km_estimate = np.array(len(y_test_time)*[float(time_bins[time_idx])])
            km_mae = mean_error(km_estimate, event_times=y_test_time, event_indicators=y_test_event,
                                train_event_times=y_train_time, train_event_indicators=y_train_event,
                                method='Margin')
            
            metrics = [ci, ibs, mae_margin, km_mae, d_calib]
            print(f'{model_name} E{event_id+1}: ' + str(metrics))
            res_sr = pd.Series([dataset_name, model_name, seed, event_id+1] + metrics,
                                index=result_cols)
    
            # Save results
            filename = f"{cfg.RESULTS_DIR}/model_results.csv"
            if os.path.exists(filename):
                results = pd.read_csv(filename)
            else:
                results = pd.DataFrame(columns=result_cols)
            results = pd.concat([results, res_sr.to_frame().T], ignore_index=True)
            results.to_csv(filename, index=False)
                