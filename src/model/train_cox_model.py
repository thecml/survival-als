import pandas as pd
import numpy as np
import config as cfg
from tools.data_loader import DataLoader
from utility.survival import split_time_event, make_stratified_split, convert_to_structured
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

if __name__ == "__main__":
    # Load data
    dl = DataLoader().load_data(event="Speech")
    num_features, cat_features = dl.get_features()
    df = dl.get_data()
    df = df.dropna(subset='DiseaseProgressionRate')

    # Split data in train/test sets
    df_train, _, df_test = make_stratified_split(df, stratify_colname='both', frac_train=0.7,
                                                 frac_valid=0, frac_test=0.3, random_state=0)
    X_train = df_train[['DiseaseProgressionRate']]
    X_test = df_test[['DiseaseProgressionRate']]
    y_train = convert_to_structured(df_train["time"], df_train["event"])
    y_test = convert_to_structured(df_test["time"], df_test["event"])
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Calculate event times
    lower, upper = np.percentile(y_train["time"], [10, 90])
    times = np.arange(lower, upper+1)
    
    # Train model
    model = CoxPHSurvivalAnalysis(alpha=0.0001)
    model.fit(X_train, y_train)
    print(model.coef_)
    
    # Evaluate
    y_pred = model.predict(X_test)
    c_harrell = concordance_index_censored(y_test["event"], y_test["time"], y_pred)
    print(c_harrell)

    
    