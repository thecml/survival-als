import pandas as pd
import numpy as np
import config as cfg
from tools.event_loader import EventDataLoader
from utility.survival import make_stratified_split, convert_to_structured, make_time_bins, make_event_times
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from utility.training import scale_data
from tools.evaluator import LifelinesEvaluator
from tools.models import CoxPH, train_model, make_cox_prediction
from tools.preprocessor import Preprocessor
from utility.training import split_and_scale_data
from torch.utils.data import DataLoader, TensorDataset
import torch
import random
import warnings

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

if __name__ == "__main__":
    # Load data
    events = ['Speech', 'Walking']
    for event in events:
        dl = EventDataLoader().load_data(event=event)
        num_features, cat_features = dl.get_features()
        df = dl.get_data()

        # Split data in train/valid/test sets
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='both', frac_train=0.7,
                                                            frac_valid=0.1, frac_test=0.2, random_state=0)
        X_train = df_train[cat_features+num_features]
        X_valid = df_valid[cat_features+num_features]
        X_test = df_test[cat_features+num_features]
        y_train = convert_to_structured(df_train["time"], df_train["event"])
        y_valid = convert_to_structured(df_valid["time"], df_valid["event"])
        y_test = convert_to_structured(df_test["time"], df_test["event"])
        
        # Scale data
        preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat='mean')
        transformer = preprocessor.fit(X_train, cat_feats=cat_features, num_feats=num_features,
                                       one_hot=True, fill_value=-1)
        X_train = transformer.transform(X_train)
        X_valid = transformer.transform(X_valid)
        X_test = transformer.transform(X_test)

        # Make event times
        time_bins = make_time_bins(y_train["time"], event=y_train["event"])

        # Format data for training the NN
        data_train = X_train.copy()
        data_train["time"] = pd.Series(y_train['time'])
        data_train["event"] = pd.Series(y_train['event']).astype(int)
        data_valid = X_valid.copy()
        data_valid["time"] = pd.Series(y_valid['time'])
        data_valid["event"] = pd.Series(y_valid['event']).astype(int)
        data_test = X_test.copy()
        data_test["time"] = pd.Series(y_test['time'])
        data_test["event"] = pd.Series(y_test['event']).astype(int)
        
        # Make data loader
        #train_dl, valid_dl, test_dl = make_dataloader(data_train, data_valid, data_test, device)
        
        # Train model
        config = dotdict(cfg.PARAMS_COX)
        n_features = X_train.shape[1]
        model = CoxPH(in_features=n_features, config=config)
        model = train_model(model, data_train, time_bins, config=config,
                            random_state=0, reset_model=True, device=device)
        
        # Evaluate
        x_test = torch.tensor(data_test.drop(["time", "event"], axis=1).values, dtype=torch.float, device=device)
        survival_outputs, time_bins, ensemble_outputs = make_cox_prediction(model, x_test, config=config)
        survival_outputs = pd.DataFrame(survival_outputs, columns=np.array(time_bins))
        lifelines_eval = LifelinesEvaluator(survival_outputs.T, y_test["time"], y_test["event"],
                                            y_train['time'], y_train['event'])
        mae_hinge = lifelines_eval.mae(method="Hinge")
        ci = lifelines_eval.concordance()[0]
        d_calib = lifelines_eval.d_calibration()[0]
        
        print(f"Evaluated {event}: CI={round(ci,2)}, MAE={round(mae_hinge,2)}, D-Calib={round(d_calib,2)}")
    