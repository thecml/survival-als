import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from scipy.interpolate import interp1d
from typing import Union
import argparse
from typing import List, Tuple
import pandas as pd

from bnn.base_layers import BayesianLinear, BayesianElementwiseLinear, BayesianHorseshoeLayer
from mensa.loss import conditional_weibull_loss_multi
from bnn.utility import mensa_survival

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def create_representation(input_dim: int, layers: List[int],
                          activation: str, config, 
                          bias: bool = True) -> nn.Sequential:
    if activation == 'ReLU6':
        act = nn.ReLU6()
    elif activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'SeLU':
        act = nn.SELU()
    elif activation == 'Tanh':
        act = nn.Tanh()

    modules = []
    prevdim = input_dim

    for hidden in layers:
        modules.append(BayesianLinear(prevdim, hidden, config, bias=bias))
        modules.append(act)
        prevdim = hidden

    return nn.Sequential(*modules)

class BayesianBaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def reset_parameters(self):
        pass

    @abstractmethod
    def log_prior(self):
        pass

    @abstractmethod
    def log_variational_posterior(self):
        pass

    def get_name(self):
        return self._get_name()

class BayesianMensa(BayesianBaseModel):
    """
    Multi-event network for survival analysis with Bayesian layers.
    """
    def __init__(self, in_features: int, n_dists: int, layers: List[int],
                 n_time_bins: int, n_events, config: argparse.Namespace):
        """Initialises the module.

        Parameters
        ----------
        in_features
            Number of input features.
        num_time_bins
            The number of bins to divide the time axis into.
        config
            Configuration/hyper-parameters of the network.
        """
        super().__init__()
        if n_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        self.config = config
        
        self.n_events = n_events
        self.n_dists = n_dists
        self.temp = 1000
        self.discount = 1.0
        
        self.in_features = in_features
        self.num_time_bins = n_time_bins

        if layers is None:
            layers = []
        self.layers = layers

        if len(layers) == 0:
            lastdim = in_features
        else:
            lastdim = layers[-1]
        
        self.act = nn.SELU()
        self.shape = nn.Parameter(-torch.ones(self.n_dists * n_events))
        self.scale = nn.Parameter(-torch.ones(self.n_dists * n_events))
        
        self.gate = BayesianLinear(in_features, self.n_dists * self.n_events, config, bias=False)
        self.scaleg = BayesianLinear(in_features, self.n_dists * self.n_events, config, bias=True)
        self.shapeg = BayesianLinear(in_features, self.n_dists * self.n_events, config, bias=True)
        
        self.embedding = BayesianElementwiseLinear(in_features, config)

    def forward(self, x: torch.Tensor, sample: bool, n_samples: int) -> torch.Tensor:
        xrep = F.relu6(self.embedding(x, n_samples=n_samples))
        dim = x.shape[0]
        shape = self.act(self.shapeg(xrep, sample, n_samples)) + self.shape.expand(dim, -1)
        scale = self.act(self.scaleg(xrep, sample, n_samples)) + self.scale.expand(dim, -1)
        gate = self.gate(xrep, sample, n_samples) / self.temp
        outcomes = []
        for i in range(self.n_events):
            outcomes.append((shape[:,:,i*self.n_dists:(i+1)*self.n_dists],
                             scale[:,:,i*self.n_dists:(i+1)*self.n_dists],
                             gate[:,:,i*self.n_dists:(i+1)*self.n_dists]))
        return outcomes

    def compute_risks_multi(self, params, ti):
        f_risks = []
        s_risks = []
        for i in range(self.n_events):
            t = ti[:,i].reshape(-1,1).expand(-1, self.n_dists)
            k = params[i][0]
            b = params[i][1]
            gate = nn.LogSoftmax(dim=1)(params[i][2])
            s = - (torch.pow(torch.exp(b)*t, torch.exp(k)))
            f = k + b + ((torch.exp(k)-1)*(b+torch.log(t)))
            f = f + s
            s = (s + gate)
            s = torch.logsumexp(s, dim=1)
            f = (f + gate)
            f = torch.logsumexp(f, dim=1)
            f_risks.append(f)
            s_risks.append(s)
        f = torch.stack(f_risks, dim=1)
        s = torch.stack(s_risks, dim=1)
        return f, s

    def log_prior(self) -> torch.Tensor:
        """
        Calculates the logarithm of the current
        value of the prior distribution over the weights
        """
        return self.embedding.log_prior + self.gate.log_prior \
               + self.scaleg.log_prior + self.shapeg.log_prior

    def log_variational_posterior(self) -> torch.Tensor:
        """
        Calculates the logarithm of the current value
        of the variational posterior distribution over the weights
        """
        return self.embedding.log_variational_posterior + self.gate.log_variational_posterior \
               + self.scaleg.log_variational_posterior + self.shapeg.log_variational_posterior

    def sample_elbo(self, x, t, e, dataset_size) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Calculate the evidence lower bound for a batch with sampling.
        :param x: covariates
        :param t: times [n_samples, k_events]
        :param e: events [n_samples, k_events]
        :param dataset_size:
        :return:
        """
        num_batch = dataset_size / self.config.batch_size
        n_samples = self.config.n_samples_train
        outputs = self.forward(x, sample=True, n_samples=n_samples) # list of K outcomes
        log_prior = self.log_prior() / n_samples
        log_variational_posterior = self.log_variational_posterior() / n_samples
        
        mean_outputs = []
        for i in range(self.n_events):
            event_output = []
            for j in range(3): # shape, scale, gate
                event_output.append(outputs[i][j].mean(dim=0)) # average samples
            mean_outputs.append(event_output)
            
        f, s = self.compute_risks_multi(mean_outputs, t)
        nll = conditional_weibull_loss_multi(f, s, e, self.n_events)
        
        # Shouldn't here be batch_size instead?
        loss = (log_variational_posterior - log_prior) / num_batch + nll
        return loss, log_prior, log_variational_posterior, nll

    def reset_parameters(self):
        """Reinitialize the model."""
        self.gate.reset_parameters()
        self.scaleg.reset_parameters()
        self.shapeg.reset_parameters()
        self.embedding.reset_parameters()
        return self

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features},"
                f" num_time_bins={self.num_time_bins})")