from typing import List
import torch
import math

def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

def relu(x, coeff):
    return torch.relu(torch.matmul(x, coeff))

class DGP_Weibull_linear: # This is PH implementation 
    def __init__(self, n_features, alpha: float, gamma: float, device="cpu", dtype=torch.float64):
        self.alpha = torch.tensor([alpha], device=device).type(dtype)
        self.gamma = torch.tensor([gamma], device=device).type(dtype)
        self.coeff = torch.rand((n_features,), device=device).type(dtype)

    def PDF(self ,t ,x):
        return self.hazard(t, x) * self.survival(t,x)
    
    def CDF(self ,t ,x):
        return 1 - self.survival(t,x)
    
    def survival(self ,t ,x):
        return torch.exp(-self.cum_hazard(t,x))
    
    def hazard(self, t, x):
        return ((self.gamma/self.alpha)*((t/self.alpha)**(self.gamma-1))) * torch.exp(torch.matmul(x, self.coeff))
        
    def cum_hazard(self, t, x):
        return ((t/self.alpha)**self.gamma) * torch.exp(torch.matmul(x, self.coeff))

    def parameters(self):
        return [self.alpha, self.gamma, self.coeff]
    
    def rvs(self, x, u):
        return ((-LOG(u)/torch.exp(torch.matmul(x, self.coeff)))**(1/self.gamma))*self.alpha

class DGP_Weibull_nonlinear: # This is nonlinear PH implementation
    def __init__(self, n_features, alpha, gamma, risk_function=relu, device="cpu", dtype=torch.float64):
        self.nf = n_features
        self.alpha = torch.tensor([alpha], device=device).type(dtype)
        self.gamma = torch.tensor([gamma], device=device).type(dtype)
        self.coeff = torch.rand((n_features,), device=device).type(dtype)
        self.risk_function = risk_function
        
    def PDF(self ,t ,x):
        return self.hazard(t, x) * self.survival(t, x)
    
    def CDF(self ,t ,x):    
        return 1 - self.survival(t, x)
    
    def survival(self ,t ,x):   
        return torch.exp(-self.cum_hazard(t, x))
    
    def hazard(self, t, x):
        return ((self.gamma/self.alpha)*((t/self.alpha)**(self.gamma-1))) * torch.exp(self.risk_function(x,self.coeff))

    def cum_hazard(self, t, x):
        return ((t/self.alpha)**self.gamma) * torch.exp(self.risk_function(x, self.coeff))
    
    def rvs(self, x, u):
        return ((-LOG(u)/torch.exp(self.risk_function(x, self.coeff)))**(1/self.gamma))*self.alpha