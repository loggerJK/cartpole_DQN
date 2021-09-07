import gym
from DQNTrainer import DQNTrainer

env = gym.make("CartPole-v0")

trainer = DQNTrainer(env)

trainer.train()
