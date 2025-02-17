import torch
from dimensions import Dimensions

class Env:
    
    def __init__(self, actions_dim, observations_dim) -> None:
        self.dimensions = Dimensions(actions_dims=actions_dim, observations_dims=observations_dim)
    
    def get_observations(self) -> torch.Tensor:
        return torch.rand(size=(self.dimensions.observations_dims,))
    
    def apply_actions(self, actions: torch.Tensor) -> None:
        return
    
    def get_reward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.rand(size=(1,))
    
    def step(self, action: torch.Tensor) -> torch.Tensor:
        return torch.rand(size=(self.dimensions.observations_dims,))