from pathlib import Path
from typing import List, Tuple, Union
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import trange
import config as cfg

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def train_model(model: nn.Module, train_dict: dict, valid_dict: dict,
                time_bins: torch.Tensor, config: dict, random_state: int,
                reset_model: bool, device: torch.device):
    train_size = train_dict['X'].shape[0]
    valid_size = valid_dict['X'].shape[0]
    
    optim_dict = [{'params': model.parameters(), 'lr': config.lr}]
    optimizer = torch.optim.Adam(optim_dict)
    
    if reset_model:
        model.reset_parameters()
    
    train_loader = DataLoader(TensorDataset(train_dict['X'].to(device),
                                            train_dict['T'].to(device),
                                            train_dict['E'].to(device)),
                                batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_dict['X'].to(device),
                                            valid_dict['T'].to(device),
                                            valid_dict['E'].to(device)),
                                batch_size=config.batch_size, shuffle=False)

    model = model.to(device)
    min_delta = 0.001
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    
    pbar = trange(config.n_epochs, disable=not config.verbose)
    
    for i in pbar:
        total_train_loss, total_valid_loss = 0, 0
        total_train_log_likelihood, total_valid_log_likelihood = 0, 0
        total_kl_divergence = 0
        
        model.train()
        for xi, ti, ei in train_loader:
            optimizer.zero_grad()
            loss, log_prior, log_variational_posterior, log_likelihood = model.sample_elbo(xi, ti, ei, train_size)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() / train_size
            total_train_log_likelihood += log_likelihood.item() / train_size
            total_kl_divergence += (log_variational_posterior.item() - log_prior.item()) * config.batch_size / train_size**2

        model.eval()
        for xi, ti, ei in valid_loader:
            valid_loss, _, _, valid_log_likelihood = model.sample_elbo(xi, ti, ei, valid_size)
            
            total_valid_loss += valid_loss.item() / valid_size
            total_valid_log_likelihood += valid_log_likelihood.item() / valid_size
        
        pbar.set_description(f"[epoch {i + 1: 4}/{config.n_epochs}]")
        pbar.set_postfix_str(f"Train: Total = {total_train_loss:.4f}, "
                             f"KL = {total_kl_divergence:.4f}, "
                             f"nll = {total_train_log_likelihood:.4f}; "
                             f"Val: Total = {total_valid_loss:.4f}, "
                             f"nll = {total_valid_log_likelihood:.4f}; ")
        if config.early_stop:
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                best_ep = i
                torch.save(model.state_dict(), Path.joinpath(cfg.MODELS_DIR, 'model.pth'))
            if (i - best_ep) > config.patience:
                print(f"Validation loss converges at {best_ep}-th epoch.")
                break
    
    model.load_state_dict(torch.load(Path.joinpath(cfg.MODELS_DIR, 'model.pth')))
    return model

def make_ensemble_mensa_prediction(
        model: torch.nn.Module,
        x: torch.Tensor,
        time_bins: NumericArrayLike,
        risk: int,
        n_dists: int,
        config: dict
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    model.eval()

    with torch.no_grad():
        # ensemble_output should have size: n_samples * dataset_size * n_bin
        t = list(time_bins.cpu().numpy())
        logits_outputs = model.forward(x, sample=True, n_samples=config.n_samples_test)
        
        survival_outputs = []
        n_events = len(logits_outputs)
        for j in range(config.n_samples_test):
            shapes = logits_outputs[risk][0][j]
            scales = logits_outputs[risk][1][j]
            logits =  logits_outputs[risk][2][j]
            params = [shapes, scales, logits]
            survival_outputs.append(torch.Tensor(mensa_survival(params, t, time_bins=time_bins, n_dists=n_dists)))
            
        survival_outputs = torch.stack(survival_outputs, dim=0)
        mean_survival_outputs = survival_outputs.mean(dim=0)

    return mean_survival_outputs, time_bins, survival_outputs

def mensa_survival(survival_outputs: List[torch.Tensor],
                   t: torch.Tensor,
                   time_bins: torch.Tensor,
                   n_dists: int):
    """
    Generates predicted survival curves from predicted logits.
    """
    shape, scale, logits = survival_outputs[0], survival_outputs[1], survival_outputs[2]
    k_ = shape
    b_ = scale

    squish = nn.LogSoftmax(dim=1)
    logits = squish(logits)
    
    t_horz = time_bins.clone().detach().double().to(logits.device)
    t_horz = t_horz.repeat(shape.shape[0], 1)
    
    cdfs = []
    for j in range(len(time_bins)):

        t = t_horz[:, j]
        lcdfs = []

        for g in range(n_dists):

            k = k_[:, g]
            b = b_[:, g]
            s = - (torch.pow(torch.exp(b)*t, torch.exp(k)))
            lcdfs.append(s)

        lcdfs = torch.stack(lcdfs, dim=1)
        lcdfs = lcdfs+logits
        lcdfs = torch.logsumexp(lcdfs, dim=1)
        cdfs.append(lcdfs.detach().cpu().numpy())
    
    return np.exp(np.array(cdfs)).T
