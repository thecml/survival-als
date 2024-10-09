import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from typing import List, Tuple, Union
from datetime import datetime
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from utility.survival import cox_survival, cox_nll

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def train_deepsurv_model(
        model: nn.Module,
        data_train: pd.DataFrame,
        data_valid: pd.DataFrame,
        time_bins: NumericArrayLike,
        config: argparse.Namespace,
        random_state: int,
        reset_model: bool = True,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float64
) -> nn.Module:
    if config.verbose:
        print(f"Training {model.get_name()}: reset mode is {reset_model}, number of epochs is {config.num_epochs}, "
              f"learning rate is {config.lr}, C1 is {config.c1}, "
              f"batch size is {config.batch_size}, device is {device}.")
    train_size = data_train.shape[0]
    val_size = data_valid.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if reset_model:
        model.reset_parameters()

    model = model.to(device)
    model.train()
    best_val_nll = np.inf
    best_ep = -1

    pbar = trange(config.num_epochs, disable=not config.verbose)

    start_time = datetime.now()
    x_train, t_train, e_train = (torch.tensor(data_train.drop(["time", "event"], axis=1).values, dtype=dtype),
                                 torch.tensor(data_train["time"].values, dtype=dtype),
                                 torch.tensor(data_train["event"].values, dtype=dtype))
    x_val, t_val, e_val = (torch.tensor(data_valid.drop(["time", "event"], axis=1).values, dtype=dtype).to(device),
                           torch.tensor(data_valid["time"].values, dtype=dtype).to(device),
                           torch.tensor(data_valid["event"].values, dtype=dtype).to(device))

    train_loader = DataLoader(TensorDataset(x_train, t_train, e_train), batch_size=train_size, shuffle=True)
    model.config.batch_size = train_size

    for i in pbar:
        nll_loss = 0
        for xi, ti, ei in train_loader:
            xi, ti, ei = xi.to(device), ti.to(device), ei.to(device)
            optimizer.zero_grad()
            y_pred = model.forward(xi)
            nll_loss = cox_nll(y_pred, 1, 0, ti, ei, model, C1=config.c1)

            nll_loss.backward()
            optimizer.step()
            # here should have only one iteration
        logits_outputs = model.forward(x_val)
        eval_nll = cox_nll(logits_outputs, 1, 0, t_val, e_val, model, C1=0)
        pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
        pbar.set_postfix_str(f"nll-loss = {nll_loss.item():.4f}; "
                                f"Validation nll = {eval_nll.item():.4f};")
        if config.early_stop:
            if best_val_nll > eval_nll:
                best_val_nll = eval_nll
                best_ep = i
            if (i - best_ep) > config.patience:
                print(f"Validation loss converges at {best_ep}-th epoch.")
                break

    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training time: {training_time.total_seconds()}")
    # model.eval()
    model.calculate_baseline_survival(x_train.to(device), t_train.to(device), e_train.to(device))
    return model

def make_deepsurv_prediction(
        model: nn.Module,
        x: torch.Tensor,
        config: argparse.Namespace,
        dtype: torch.dtype
):
    model.eval()
    start_time = datetime.now()
    with torch.no_grad():
        pred = model.forward(x)
        end_time = datetime.now()
        inference_time = end_time - start_time
        if config.verbose:
            print(f"Inference time: {inference_time.total_seconds()}")
        survival_curves = cox_survival(model.baseline_survival, pred, dtype)
        survival_curves = survival_curves.squeeze()

    time_bins = model.time_bins
    return survival_curves, time_bins