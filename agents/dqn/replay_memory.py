from collections import deque
from typing import List, Tuple

import numpy as np


class ReplayMemory:
    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.memory = deque()

    def add(self, observation, action, reward, terminal) -> None:
        while len(self.memory) >= memory_size:
            self.memory.popleft()

        self.memory.append((observation, action, reward, terminal))

    def sample(self, batch_size: int) -> List[Tuple]:
        indices = np.random.randint(0, len(self.observations), batch_size)
        return [self.memory[i] for i in indices]
