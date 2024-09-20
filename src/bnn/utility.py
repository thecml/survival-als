import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import trange

def train_model(model: nn.Module, train_dict: dict, valid_dict: dict,
                time_bins: torch.Tensor, config: dict, random_state: int,
                reset_model: bool, device: torch.device):
    train_size = train_dict['X'].shape[0]
    valid_size = valid_dict['X'].shape[0]
    
    optim_dict = [{'params': model.parameters(), 'lr': config.lr, "weight_decay":1e-5}]
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    
    return model

def mensa_survival(logits: torch.Tensor,
                   t: torch.Tensor,
                   time_bins: torch.Tensor,
                   n_dists: int,
                   risk: int,
                   with_sample: bool = True):
    """Generates predicted survival curves from predicted logits.
    """
    if with_sample:
        assert logits.dim() == 3, "The logits should have dimension with with size (n_samples, n_data, n_bins)"
        raise NotImplementedError()
    else: # no sampling
        assert logits.dim() == 2, "The logits should have dimension with with size (n_data, n_bins)"
        shape, scale, logits = logits[risk][0], logits[risk][1], logits[risk][2]
        k_ = shape
        b_ = scale

        squish = nn.LogSoftmax(dim=1)
        logits = squish(logits)
        
        t_horz = torch.tensor(time_bins).double().to(logits.device)
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
        
        