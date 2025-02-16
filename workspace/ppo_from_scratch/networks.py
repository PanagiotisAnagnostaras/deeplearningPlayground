from torch import nn
import torch
from torch.distributions import MultivariateNormal

class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim)
        )
        cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
    
    def forward(self, x):
        return self.linear_relu_stack(x)
    
    def get_action(self, obs):
        mean = self(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

class Critic(nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.linear_relu_stack(x)
