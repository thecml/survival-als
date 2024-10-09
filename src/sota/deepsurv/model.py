import argparse
import torch
import torch.nn as nn
from utility.survival import calculate_baseline_hazard

class DeepSurv(nn.Module):
    def __init__(self, in_features: int, config: argparse.Namespace):
        super().__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        n_hidden = 100
        
        # Shared parameters
        self.hidden = nn.Sequential(
            nn.Linear(in_features, n_hidden),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(n_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared embedding
        hidden = self.hidden(x)
        return self.fc1(hidden)

    def calculate_baseline_survival(self, x, t, e):
        outputs = self.forward(x)
        self.time_bins, self.cum_baseline_azhard, self.baseline_survival = calculate_baseline_hazard(outputs, t, e)

    def reset_parameters(self):
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}"

    def get_name(self):
        return self._get_name()