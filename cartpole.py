import gym
from ReplayMemory import ReplayMemory
import tensorflow as tf
from tensorflow import keras
import numpy as np

env = gym.make("CartPole-v0")

model = keras.models.load_model("test.h5")

# DQNAgent를 이용해 CartPole을 실행한다

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = np.argmax(model.predict(observation.reshape(-1, 4)))
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timestpes".format(t + 1))
            break
