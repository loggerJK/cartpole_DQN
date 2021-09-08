import gym
from DQNTrainer import DQNTrainer
from tensorflow import keras

env = gym.make("CartPole-v0")

trainer = DQNTrainer(env, temp_save_freq=1, epsilon=0.99, epsilon_decay=0.9999)

trainer.agent.model = keras.models.load_model("model_1.h5")

trainer.train()
