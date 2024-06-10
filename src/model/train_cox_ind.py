import pandas as pd
import numpy as np
import config as cfg
from tools.event_loader import EventDataLoader
from utility.survival import split_time_event, make_stratified_split, convert_to_structured, make_event_times
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from utility.training import scale_data
from tools.evaluator import LifelinesEvaluator

if __name__ == "__main__":
    # Load data
    events = ['Speech', 'Walking']
    for event in events:
        dl = EventDataLoader().load_data(event=event)
        num_features, cat_features = dl.get_features()
        df = dl.get_data()

        # Split data in train/test sets
        df_train, _, df_test = make_stratified_split(df, stratify_colname='both', frac_train=0.8,
                                                     frac_valid=0, frac_test=0.2, random_state=0)
        X_train = df_train[cat_features+num_features]
        X_test = df_test[cat_features+num_features]
        y_train = convert_to_structured(df_train["time"], df_train["event"])
        y_test = convert_to_structured(df_test["time"], df_test["event"])
        
        # Scale data
        X_train, X_test = scale_data(X_train, X_test, cat_features, num_features)

        # Split data in train/test sets
        t_train, e_train = split_time_event(y_train)
        t_test, e_test = split_time_event(y_test)
        
        # Make event times
        event_times = make_event_times(t_train, e_train)
        
        # Train model
        model = CoxPHSurvivalAnalysis(alpha=0.0001)
        model.fit(X_train, y_train)
        
        # Evaluate
        surv_preds = model.predict_survival_function(X_test)
        surv_preds = pd.DataFrame(np.row_stack([fn(event_times) for fn in surv_preds]), columns=event_times)
        lifelines_eval = LifelinesEvaluator(surv_preds.T, y_test["time"], y_test["event"], t_train, e_train)
        mae_hinge = lifelines_eval.mae(method="Hinge")
        ci = lifelines_eval.concordance()[0]
        d_calib = lifelines_eval.d_calibration()[0]
        
        print(f"Evaluated {event}: CI={ci}, MAE={mae_hinge}, D-Calib={d_calib}")
        
    