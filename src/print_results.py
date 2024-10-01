import pandas as pd
from pathlib import Path
import glob
import os
import config as cfg
import numpy as np

N_DECIMALS = 2
ALPHA = 0.05

def calculate_calib(df, metric_name, dataset_name, event_id):
    results = df.loc[(df['DatasetName'] == dataset_name)]
    num_seeds = df['Seed'].nunique()
    event_ratios = []
    num_calib = results.loc[results['EvenId'] == event_id][metric_name].apply(lambda x: (x > ALPHA)).sum()
    event_ratio = f"{num_calib}/{num_seeds}"
    event_ratios.append(event_ratio)
    result_string = "(" + ', '.join(event_ratios) + ")"
    return result_string

if __name__ == "__main__":
    path = Path.joinpath(cfg.RESULTS_DIR, f"results.csv")
    df = pd.read_csv(path)
    df = df.round(N_DECIMALS).fillna(0)
    
    dataset_names = ["proact"]
    model_names = ["deepsurv", 'hierarch', 'bayesian']
    event_names = ["Speech", "Salivation", "Swallowing", "Handwriting", "Walking", "Death"]
    metric_names = ["CI", "IBS", "MAEM", "GlobalCI", "LocalCI", "DCalib", "CCalib"]
    
    for dataset_name in dataset_names:
        for model_name in model_names:
            for event_id, event_name in enumerate(event_names):
                text = ""
                results = df.loc[(df['DatasetName'] == dataset_name) & (df['ModelName'] == model_name)
                                 & (df['EvenId'] == event_id+1)]
                if results.empty:
                    break
                text += f"{event_name} & "
                for i, metric_name in enumerate(metric_names):
                    metric_result = results[metric_name]
                    if metric_name  == "DCalib":
                        d_calib = calculate_calib(df, metric_name, dataset_name, event_id+1)
                        text += f"{d_calib} & "
                    elif model_name == "bayesian" and metric_name  == "CCalib":
                        c_calib = calculate_calib(df, metric_name, dataset_name, event_id+1)
                        text += f"{c_calib}"
                    else:
                        mean = f"%.{N_DECIMALS}f" % round(np.mean(metric_result), N_DECIMALS)
                        std = f"%.{N_DECIMALS}f" % round(np.std(metric_result), N_DECIMALS)
                        text += f"{mean}$\pm${std} & "
                text += " \\\\"
                print(text)
            print()
            break