from ppo import PPO
from env import Env

if __name__=="__main__":
    env = Env(actions_dim=1, observations_dim=4)
    ppo = PPO(env=env)
    ppo.train(total_training_steps=100)
    ppo.save()
    