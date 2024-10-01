import pandas as pd
import numpy as np
import torch
import config as cfg
from utility.survival import (split_time_event, make_stratified_split,
                              convert_to_structured, make_event_times, preprocess_data)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from tools.data_loader import get_data_loader
from SurvivalEVAL.Evaluator import LifelinesEvaluator
from SurvivalEVAL.Evaluations.util import KaplanMeier
from SurvivalEVAL import mean_error

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
    X_train = pd.DataFrame(train_dict['X'], columns=dl.columns)
    X_valid = pd.DataFrame(valid_dict['X'], columns=dl.columns)
    X_test = pd.DataFrame(test_dict['X'], columns=dl.columns)
    X_train, X_valid, X_test = preprocess_data(X_train, X_valid, X_test, cat_features,
                                               num_features, as_array=False)
    n_features = X_train.shape[1]
    
    events = ['Speech', 'Salivation', 'Swallowing', "Handwriting", "Walking", 'Death']
    for i, event in enumerate(events):
        y_train = convert_to_structured(train_dict['T'][:,i], train_dict['E'][:,i])
        y_valid = convert_to_structured(valid_dict['T'][:,i], valid_dict['E'][:,i])
        y_test = convert_to_structured(test_dict['T'][:,i], test_dict['E'][:,i])
    
        # Train model
        model = CoxPHSurvivalAnalysis(alpha=0.0001)
        model.fit(X_train, y_train)
        
        # Evaluate
        times = model.unique_times_
        surv_preds = model.predict_survival_function(X_test)
        surv_preds = pd.DataFrame(np.row_stack([fn(times) for fn in surv_preds]), columns=times)
        surv_preds.insert(0, 0.0, 1) # add 1 to the beginning
        lifelines_eval = LifelinesEvaluator(surv_preds.T, y_test["time"], y_test["event"],
                                            y_train["time"], y_train["event"])
        mae_margin = lifelines_eval.mae(method="Margin")
        ci = lifelines_eval.concordance()[0]
        ibs = lifelines_eval.integrated_brier_score()
        d_calib = lifelines_eval.d_calibration()[0]
        
        # Calculate KM estimate
        km_model = KaplanMeier(y_train["time"], y_train["event"])
        km_surv_prob = torch.from_numpy(km_model.predict(times))
        time_idx = np.where(km_surv_prob <= 0.5, km_surv_prob, -np.inf).argmax(axis=0)
        km_estimate = np.array(len(y_test["time"])*[float(times[time_idx])])
        km_mae = mean_error(km_estimate, event_times=y_test["time"], event_indicators=y_test["event"],
                            train_event_times=y_train["time"], train_event_indicators=y_train["event"],
                            method='Margin')
        
        print(f"Evaluated E{i+1}: CI={round(ci, 3)}, IBS={round(ibs, 3)}, " +
              f"MAE={round(mae_margin, 3)}, D-Calib={round(d_calib, 3)}, " +
              f"KM MAE: {round(km_mae, 3)}")
    