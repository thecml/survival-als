import numpy as np
import pandas as pd
import torch
import config as cfg
from sksurv.util import Surv

from tools.data_loader import get_data_loader
from utility.survival import make_time_bins, preprocess_data
from SurvivalEVAL.Evaluations.util import KaplanMeier
#from tools.formatter import Formatter
#from xgbse.non_parametric import calculate_kaplan_vectorized

matplotlib_style = 'default'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
plt.rcParams.update({'axes.labelsize': 'medium',
                    'axes.titlesize': 'medium',
                    'font.size': 14.0,
                    'text.usetex': True,
                    'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{bm}'})

class _TFColor(object):
    """Enum of colors used in TF docs."""
    red = '#F15854'
    blue = '#5DA5DA'
    orange = '#FAA43A'
    green = '#60BD68'
    pink = '#F17CB0'
    brown = '#B2912F'
    purple = '#B276B2'
    yellow = '#DECF3F'
    gray = '#4D4D4D'
    def __getitem__(self, i):
        return [
            self.red,
            self.orange,
            self.green,
            self.blue,
            self.pink,
            self.brown,
            self.purple,
            self.yellow,
            self.gray,
        ][i % 9]
TFColor = _TFColor()

# Setup precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # Load data
    dl = get_data_loader("proact").load_data()
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

    # Make time bins
    time_bins = make_time_bins(train_dict['T'].cpu(), event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))
    
    
    
    km_model = KaplanMeier(y_train_time, y_train_event)
    km_surv_prob = torch.from_numpy(km_model.predict(time_bins))
    
    km_mean, km_high, km_low = calculate_kaplan_vectorized(y['Survival_time'].reshape(1,-1),
                                                            y['Event'].reshape(1,-1),
                                                            continuous_times)
    plt.plot(km_mean.columns, km_mean.iloc[0], 'k--', linewidth=2, alpha=1, label=r"$\mathbb{E}[S(t)]$ Kaplan-Meier", color="black")
    plt.fill_between(km_mean.columns, km_low.iloc[0], km_high.iloc[0], alpha=0.2, color="black")
    plt.xlabel("Time (min)")
    plt.ylabel("Survival probability S(t)")
    plt.tight_layout()
    plt.grid(alpha=0.25)
    plt.legend()
    plt.savefig(f'{cfg.PLOTS_DIR}/kaplan_meier_C{condition+1}_cens_{int(pct*100)}.pdf', format='pdf', bbox_inches="tight")
    plt.close()
