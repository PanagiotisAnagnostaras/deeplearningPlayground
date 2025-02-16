from networks import Actor, Critic

class PPO:
    def __init__(self, env) -> None:
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.act_dim)
        self.critic = Critic(self.obs_dim)
    
    def _init_hyperparameters(self) -> None:
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        
    def learn(self, total_timesteps):
        t_so_far = 0
        while t_so_far<total_timesteps:
            pass
            t_so_far+=1
    