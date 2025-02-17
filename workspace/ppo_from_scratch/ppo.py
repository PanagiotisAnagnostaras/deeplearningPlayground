import torch.torch_version
from networks import Actor, Critic
import torch
from env import Env
from typing import List
from torch.distributions import MultivariateNormal

class PPO:
    def __init__(self, env) -> None:
        self.env: Env = env
        self._init_hyperparameters()
        self.actor = Actor(self.env.dimensions.observations_dims, self.env.dimensions.actions_dims)
        self.actor_opt = torch.optim.SGD(self.actor.parameters(), lr=self.learning_rate)
        self.critic = Critic(self.env.dimensions.observations_dims)
        self.critic_opt = torch.optim.SGD(self.critic.parameters(), lr=self.learning_rate)
    
    def _init_hyperparameters(self) -> None:
        self.timesteps_per_rollout = 10
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95
        self.lambd = 0.5
        self.n_epochs = 10
        self.epsilon = 0.2
        self.learning_rate = 1e-3
        cov_var = torch.full(size=(self.env.dimensions.actions_dims,), fill_value=0.5)
        self.cov_mat = torch.diag(cov_var)

    def train(self, total_timesteps):
        t_so_far = 0
        while t_so_far<total_timesteps:
            obs, acts, rews, log_prob = self.rollout()
            advantages, rews2go = self.compute_advantages_and_rews2go(rews=rews, obs=obs)
            for epoch in range(self.n_epochs):
                self.update_actor(log_prob_before=log_prob, obs=obs, advantages=advantages)
                self.update_critic(obs=obs, rews2go=rews2go)
            t_so_far+=1
    
    def rollout(self):
        obs = []
        acts = []
        rews = []
        log_probs = []
        s = self.env.get_observations()
        obs.append(s)
        for t in range(self.timesteps_per_rollout):
            a, log_prob = self.compute_actions(s)
            s = self.env.step(a)
            r = self.env.get_reward(s, a)
            acts.append(a)
            obs.append(s)
            rews.append(r)
            log_probs.append(log_prob)
        return torch.stack(obs), torch.stack(acts), torch.stack(rews), torch.stack(log_probs)
        
    def compute_advantages_and_rews2go(self, rews: List[float], obs: List[torch.Tensor]) -> torch.Tensor:
        advantages = []
        rews2go = []
        for rew_id, rew in reversed(list(enumerate(rews))):
            next_observation = obs[rew_id+1]
            observation = obs[rew_id]
            delta = rew + self.gamma * self.read_value_function(observation) - self.read_value_function(next_observation)
            advantage = delta + (self.gamma * self.lambd) * advantages[0] if len(advantages)!=0 else delta
            advantages.insert(0, advantage)
            rew2go = rew + self.gamma * (rews2go[0]) if len(rews2go)!=0 else rew
            rews2go.insert(0, rew2go)
        return torch.stack(advantages), torch.stack(rews2go)
            
    
    def update_actor(self, log_prob_before, obs, advantages) -> None:
        _, log_prob = self.compute_actions(obs=obs)
        ratio = torch.exp(log_prob[:-1] - log_prob_before)
        surr_clipped = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
        surr_unclipped = ratio * advantages
        actor_loss = torch.min(surr_clipped, surr_unclipped).mean()
        self.actor.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_opt.step()
        print(f"actor loss = {actor_loss}")
    
    def update_critic(self, obs, rews2go) -> None:
        predicted_V = self.read_value_function(obs=obs[:-1,:])
        critic_loss = torch.nn.MSELoss()(predicted_V, rews2go)
        self.critic_opt.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_opt.step()
        print(f"critic loss = {critic_loss}")
    
    def read_value_function(self, obs: torch.Tensor) -> float:
        return self.critic(obs)
    
    def compute_actions(self, obs: torch.Tensor) -> torch.Tensor:
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach(), log_prob.detach()
    
    def save(self) -> None:
        pass