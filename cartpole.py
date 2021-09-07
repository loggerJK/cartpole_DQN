import gym
from ReplayMemory import ReplayMemory

env = gym.make("CartPole-v0")


# 에피소드를 수행하면서 얻은 experiences를 ReplayMemory에 저장한다

# 수행할 게임(=에피소드)의 수
num_episodes = 20
# 게임(에피소드)마다 수행할 최대 step
max_step = 100
replayMemory = ReplayMemory()

for episode in range(num_episodes):
    state = env.reset()
    for step in range(max_step):
        # 랜덤하게 action을 선택한다
        action = env.action_space.sample()  # 0 or 1
        # action을 수행한다
        next_state, reward, done, info = env.step(action)
        # 리플레이 메모리에 저장한다
        replayMemory.add([state, next_state, reward, done, info])
        # state를 업데이트한다
        next_state = state


# ReplayMemory를 이용해 DQNAgent를 훈련시킨다

# DQNAgent를 이용해 CartPole을 실행한다

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timestpes".format(t + 1))
            break
