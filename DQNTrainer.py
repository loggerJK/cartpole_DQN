from DQNAgent import DQNAgent
import numpy as np
import gym
from tqdm import tqdm


class DQNTrainer(object):
    def __init__(
        self,
        env: gym.Env,
        max_episode=30000,
        step_size=2000,
        epsilon=0.99,
        temp_save_freq=10,
        model_path="model",
        version="test",
        min_epsilon=0.1,
        epsilon_decay=0.99,
    ):
        self.agent = DQNAgent()
        self.max_episode = max_episode
        self.env = env
        self.step_size = step_size
        self.epsilon = epsilon
        # 학습할때 임시로 저장할 빈도
        self.temp_save_freq = temp_save_freq
        self.model_path = model_path
        self.version = version
        self.target_model_path = "target_" + model_path
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

    def train(self):
        pbar = tqdm(initial=0, total=self.max_episode, unit="episodes")
        for episode in range(self.max_episode):
            cur_state = self.env.reset()
            for step in range(self.step_size):
                if (np.random.randn(1)) <= self.epsilon:
                    # 0, 1 중에서 무작위로 수를 하나 뽑는다
                    action = np.random.randint(2)
                else:
                    # Q(cur_state,a)중에서 가장 값이 높도록 하는 a를 action으로 고른다
                    output = self.agent.forward(cur_state.reshape(-1, 4))
                    output = np.argmax(output)
                    action = output

                next_state, reward, done, info = self.env.step(action)

                # replayMemory에 결과를 저장한다
                self.agent.replayMemory.add(
                    (cur_state, action, reward, done, info, next_state)
                )

                # replayMemory를 이용해 학습을 진행한다
                self.agent.train()

                if done:
                    break

            # target_model의 가중치를 model과 동기화
            self.agent._update_target_model()

            # epsilon decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

            # 설정한 빈도에 따라서 임시 저장
            if (episode % self.temp_save_freq) == 0:
                self.agent.save(
                    self.model_path, self.target_model_path, str(self.temp_save_freq)
                )

            pbar.update(1)

        # 모든 학습이 끝나면 모델을 저장한다
        self.agent.save(self.model_path, self.target_model_path, self.version)
