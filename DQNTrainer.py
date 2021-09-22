from sys import version
from DQNAgent import DQNAgent
import numpy as np
import gym
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


class DQNTrainer(object):
    def __init__(
        self,
        env: gym.Env,
        max_episode=500,
        step_size=2000,
        epsilon=0.99,
        epsilon_decay=0.995,
        temp_save_freq=100,
        model_path=os.getcwd(),
        model_name="model",
        version="test",
        min_epsilon=0.01,
        save_on_colab=False,
    ):
        self.agent = DQNAgent()
        self.max_episode = max_episode
        self.env = env
        self.step_size = step_size
        self.epsilon = epsilon
        # 학습할때 임시로 저장할 빈도
        self.temp_save_freq = temp_save_freq
        # 모델을 저장할 경로
        self.model_path = model_path
        # 저장할 모델의 경로
        self.model_name = model_name
        # 저장할 최종 모델의 버전
        self.version = version
        self.target_model_path = "target_" + model_path
        # epsilon greedy
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        # colab에서 저장할지 여부
        self.save_on_colab = save_on_colab
        self.save_epi_reward = []

    def train(self):
        pbar = tqdm(initial=0, total=self.max_episode, unit="episodes")

        for episode in range(self.max_episode):
            cur_state = self.env.reset()
            episode_reward, done = 0, False

            while not done:
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

                # 상태 업데이트
                cur_state = next_state
                episode_reward += reward

                if done:
                    break

            # target_model의 가중치를 model과 동기화
            self.agent._update_target_model()

            # epsilon decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

            # 설정한 빈도에 따라서 임시 저장
            if (episode % self.temp_save_freq) == 0:
                if self.save_on_colab:
                    self.colab_save(
                        model_name="model", version=self.version, num_trained=episode
                    )
                else:
                    self.agent.save(
                        path=self.model_path,
                        model_name="model",
                        version=self.version,
                        num_trained=episode,
                    )

            pbar.update(1)

            ####### 한 EPISODE 종료 #########

            self.save_epi_reward.append(episode_reward)

        # --------------모든 에피소드 종료---------------- #

        # 모든 학습이 끝나면 모델을 저장한다
        self.agent.save(
            self.model_path,
            self.target_model_path,
            self.version,
            num_trained=self.max_episode,
        )

        # episode에 따른 학습결과 (reward의 총합)을 그래프로 표시한다.
        plt.plot(self.save_epi_reward)

    def colab_save(self, model_name: str, version: str, num_trained: int):
        from google.colab import drive
        import os

        mount_path = "/content/drive"
        drive.mount(mount_path)

        model_path = os.path.join(mount_path, "MyDrive", "model")

        # save model on google drive
        self.agent.save(
            path=model_path,
            model_name=model_name,
            version=version,
            num_trained=num_trained,
        )

    def colab_load(self, model_name: str, version: str, num_trained: int):
        from google.colab import drive
        import os

        mount_path = "/content/drive"
        drive.mount(mount_path)

        model_path = os.path.join(mount_path, "MyDrive", "model")

        # load model on google drive
        self.agent.load(
            path=model_path,
            model_name=model_name,
            version=version,
            num_trained=num_trained,
        )
