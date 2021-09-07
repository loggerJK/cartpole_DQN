import gym
from DQNTrainer import DQNTrainer

env = gym.make("CartPole-v0")

trainer = DQNTrainer(env, temp_save_freq=1)

trainer.train()
