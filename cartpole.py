import gym
from ReplayMemory import ReplayMemory
import tensorflow as tf
from tensorflow import keras
import numpy as np

env = gym.make("CartPole-v0")

model = keras.models.load_model("model_version.h5")

# DQNAgent를 이용해 CartPole을 실행한다

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        predicted = model.predict(observation.reshape(-1, 4))
        action = np.argmax(predicted)
        if action == 0:
            action = 1
        else:
            action = 0
        print(f"predicted = {predicted}, action = {action}")
        # action = np.random.randint(2)
        observation, reward, done, info = env.step(action)
        if done:
            print(f"{i_episode+1}th episode is finished after {t+1} timesteps")
            break
