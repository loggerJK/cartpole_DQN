from tensorflow import keras
from keras import models
from keras import layers
from tensorflow.python.keras.backend import relu
from tensorflow.python.keras.models import Sequential
from ReplayMemory import ReplayMemory
import numpy as np
from ReplayMemory import ReplayMemory


class DQNAgent(object):
    def __init__(self, batch_size=100, gamma=0.9):
        # 학습에 사용할 model과 target_model을 설정한다
        self.model = self._create_model()
        self.target_model = self._create_model()
        # 처음에는 두 모델을 동일 weight로 설정해준다
        self.target_model.set_weights(self.model.get_weights())
        self.replayMemory = ReplayMemory()
        self.gamma = gamma  # 감마, 클수록 미래의 이익을 고려한다
        self.batch_size = batch_size
        self.callbacks = [
            keras.callbacks.TensorBoard(
                log_dir="my_log_dir",
                histogram_freq=1,
                embeddings_freq=1,
            )
        ]

    def _create_model(self) -> Sequential:
        model = models.Sequential()
        model.add(layers.Dense(30, activation=relu, input_shape=(4,)))
        model.add(layers.Dense(30, activation=relu))
        model.add(layers.Dense(2))
        model.compile(optimizer="rmsprop", loss="mse")
        return model

    def forward(self, data):
        return self.model.predict(data)

    # replayMemory를 이용해 Agent를 학습
    def train(self):

        # batch_size보다 experience가 많을 경우에만 train
        if self.batch_size > len(self.replayMemory):
            return

        # batch_size만큼 샘플링한다
        # (cur_state, action, reward, done, info, next_state) : list
        samples = self.replayMemory.sample(self.batch_size)
        # batch data를 생성한다

        current_states = np.stack([sample[0] for sample in samples])
        current_q = self.model.predict(current_states)
        next_states = np.stack([sample[5] for sample in samples])
        next_q = self.target_model.predict(next_states)

        for i, (cur_state, action, reward, done, info, next_state) in enumerate(
            samples
        ):
            if done:
                next_q_value = reward
            else:
                next_q_value = reward + self.gamma * np.max(next_q[i])
            current_q[i][action] = next_q_value

        # 학습!!
        self.model.fit(
            x=current_states,
            y=current_q,
            batch_size=self.batch_size,
            verbose=False,
            callbacks=self.callbacks,
        )

    # target model의 가중치를 model의 가중치로 update 한다
    def _update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self, model_path: str, target_model_path: str, version: str):
        self.model.save(model_path + "_" + version)
        self.target_model.save(target_model_path + "_" + version)

    def load(self, model_path: str, target_model_path: str, version: str):
        self.model = keras.models.load_model(model_path + "_" + version)
        self.target_model = keras.models.load_model(target_model_path + "_" + version)
