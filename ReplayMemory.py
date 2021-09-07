from collections import deque
import random


class ReplayMemory(object):

    # 생성자
    # ReplayMemory를 저장할 배열을 생성한다
    def __init__(self, capacity=1000):
        self.memory = deque([], maxlen=capacity)

    # 데이터를 tuple형태로 받아서 ReplayMemory에 저장한다
    def add(self, data: tuple):
        self.memory.append(data)

    # self.memory에서 batch_size만큼 랜덤하게 샘플링해서 반환한다
    def sample(self, batch_size) -> list:
        return random.sample(self.memory, k=batch_size)

    def __len__(self) -> int:
        return len(self.memory)
