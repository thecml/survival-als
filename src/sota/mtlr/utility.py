import numpy as np
import torch
import torch.nn as nn
import argparse
import pandas as pd
from typing import List, Tuple, Union
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from utility.survival import reformat_survival
from torch.optim import Adam

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def masked_logsumexp(
        x: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1
) -> torch.Tensor:
    """Computes logsumexp over elements of a tensor specified by a mask
    in a numerically stable way.

    Parameters
    ----------
    x
        The input tensor.
    mask
        A tensor with the same shape as `x` with 1s in positions that should
        be used for logsumexp computation and 0s everywhere else.
    dim
        The dimension of `x` over which logsumexp is computed. Default -1 uses
        the last dimension.

    Returns
    -------
    torch.Tensor
        Tensor containing the logsumexp of each row of `x` over `dim`.
    """
    max_val, _ = (x * mask).max(dim=dim)
    max_val = torch.clamp_min(max_val, 0)
    return torch.log(
        torch.sum(torch.exp((x - max_val.unsqueeze(dim)) * mask) * mask,
                  dim=dim)) + max_val

def mtlr_nll(
        logits: torch.Tensor,
        target: torch.Tensor,
        model: torch.nn.Module,
        C1: float,
        average: bool = False
) -> torch.Tensor:
    """Computes the negative log-likelihood of a batch of model predictions.

    Parameters
    ----------
    logits : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.
    target : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the encoded ground truth survival.
    model
        PyTorch Module with at least `MTLR` layer.
    C1
        The L2 regularization strength.
    average
        Whether to compute the average log likelihood instead of sum
        (useful for minibatch training).

    Returns
    -------
    torch.Tensor
        The negative log likelihood.
    """
    censored = target.sum(dim=1) > 1
    nll_censored = masked_logsumexp(logits[censored], target[censored]).sum() if censored.any() else 0
    nll_uncensored = (logits[~censored] * target[~censored]).sum() if (~censored).any() else 0

    # the normalising constant
    norm = torch.logsumexp(logits, dim=1).sum()

    nll_total = -(nll_censored + nll_uncensored - norm)
    if average:
        nll_total = nll_total / target.size(0)

    # L2 regularization
    for k, v in model.named_parameters():
        if "mtlr_weight" in k:
            nll_total += C1/2 * torch.sum(v**2)

    return nll_total

def mtlr_survival(
        logits: torch.Tensor,
        with_sample: bool = True
) -> torch.Tensor:
    # TODO: do not reallocate G in every call
    if with_sample:
        assert logits.dim() == 3, "The logits should have dimension with with size (n_samples, n_data, n_bins)"
        G = torch.tril(torch.ones(logits.shape[2], logits.shape[2])).to(logits.device)
        density = torch.softmax(logits, dim=2)
        G_with_samples = G.expand(density.shape[0], -1, -1)

        # b: n_samples; i: n_data; j: n_bin; k: n_bin
        return torch.einsum('bij,bjk->bik', density, G_with_samples)
    else:   # no sampling
        assert logits.dim() == 2, "The logits should have dimension with with size (n_data, n_bins)"
        G = torch.tril(torch.ones(logits.shape[1], logits.shape[1])).to(logits.device)
        density = torch.softmax(logits, dim=1)
        return torch.matmul(density, G)

def make_mtlr_prediction(
        model: nn.Module,
        x: torch.Tensor,
        time_bins: NumericArrayLike,
        config: argparse.Namespace
):
    model.eval()
    start_time = datetime.now()
    with torch.no_grad():
        pred = model.forward(x)
        end_time = datetime.now()
        inference_time = end_time - start_time
        survival_curves = mtlr_survival(pred, with_sample=False)
    
    time_bins = torch.cat([torch.tensor([0], device=survival_curves.device), time_bins], dim=0).to(survival_curves.device)
    return survival_curves, time_bins, survival_curves.unsqueeze(0)

def train_mtlr_model(
        model: nn.Module,
        data_train: pd.DataFrame,
        data_val: pd.DataFrame,
        time_bins: NumericArrayLike,
        config: dotdict,
        random_state: int,
        dtype: torch.dtype,
        reset_model: bool = True,
        device: torch.device = torch.device("cuda")
) -> nn.Module:
    if config.verbose:
        print(f"Training {model.get_name()}: reset mode is {reset_model}, number of epochs is {config.num_epochs}, "
              f"learning rate is {config.lr}, C1 is {config.c1}, "
              f"batch size is {config.batch_size}, device is {device}.")
    train_size = data_train.shape[0]
    val_size = data_val.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if reset_model:
        model.reset_parameters()

    model = model.to(device)
    model.train()
    best_val_nll = np.inf
    best_ep = -1

    pbar = trange(config.num_epochs, disable=not config.verbose)

    start_time = datetime.now()
    x, y = reformat_survival(data_train, time_bins, dtype)
    x_val, y_val = reformat_survival(data_val, time_bins, dtype)
    x_val, y_val = x_val.to(device), y_val.to(device)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=config.batch_size, shuffle=True)
    for i in pbar:
        nll_loss = 0
        for xi, yi in train_loader:
            xi, yi = xi.to(device), yi.to(device)
            optimizer.zero_grad()
            y_pred = model.forward(xi)
            loss = mtlr_nll(y_pred, yi, model, C1=config.c1, average=False)

            loss.backward()
            optimizer.step()

            nll_loss += (loss / train_size).item()
        logits_outputs = model.forward(x_val)
        eval_nll = mtlr_nll(logits_outputs, y_val, model, C1=0, average=True)
        pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
        pbar.set_postfix_str(f"nll-loss = {nll_loss:.4f}; "
                                f"Validation nll = {eval_nll.item():.4f};")
        if config.early_stop:
            if best_val_nll > eval_nll:
                best_val_nll = eval_nll
                best_ep = i
            if (i - best_ep) > config.patience:
                break
    end_time = datetime.now()
    training_time = end_time - start_time
    return model