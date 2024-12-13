import pandas as pd
from pathlib import Path
import config as cfg
import numpy as np

N_DECIMALS = 2
ALPHA = 0.05

if __name__ == "__main__":
    path = Path.joinpath(cfg.RESULTS_DIR, f"model_results.csv")
    df = pd.read_csv(path)
    
    cols_to_scale = ["CI", "IBS"]
    df[cols_to_scale] = df[cols_to_scale] * 100
    
    dataset_names = ["proact"]
    model_names = ['coxph', 'rsf', 'deepsurv', 'mtlr', 'mensa']
    event_names = ['Speech', 'Swallowing', 'Handwriting', 'Walking']
    metric_names = ["CI", "IBS", "MAEM", "DCalib"]
    
    for dataset_name in dataset_names:
        for model_name in model_names:
            for event_id, event_name in enumerate(event_names):
                text = ""
                results = df.loc[(df['DatasetName'] == dataset_name)
                                 & (df['ModelName'] == model_name)
                                 & (df['EventId'] == event_id+1)]
                if results.empty:
                    break
                text += f"& {event_name} & "
                for i, metric_name in enumerate(metric_names):
                    metric_result = results[metric_name]
                    if metric_name  == "DCalib":
                        num_calib = metric_result.apply(lambda x: (x > ALPHA)).sum()
                        d_calib = f"({num_calib}/{len(metric_result)})"
                        text += f"{d_calib}"
                    else:
                        mean = f"%.{N_DECIMALS}f" % round(np.mean(metric_result), N_DECIMALS)
                        std = f"%.{N_DECIMALS}f" % round(np.std(metric_result), N_DECIMALS)
                        text += f"{mean}$\pm${std} & "
                text += " \\\\"
                print(text)
            print()