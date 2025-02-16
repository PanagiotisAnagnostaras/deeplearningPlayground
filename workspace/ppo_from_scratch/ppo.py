from networks import Actor, Critic
import torch

class PPO:
    def __init__(self, env) -> None:
        self._init_hyperparameters()
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.act_dim)
        self.critic = Critic(self.obs_dim)
    
    def _init_hyperparameters(self) -> None:
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95
        
    def learn(self, total_timesteps):
        t_so_far = 0
        while t_so_far<total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            t_so_far+=1
    
    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []
            obs = self.env.reset()
            done = False
            for ep_t in range (self.max_timesteps_per_episode):
                t += 1
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                if done:
                    break
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs
