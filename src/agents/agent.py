from abc import ABC, abstractmethod
from typing import List

import tensorflow as tf

from environments.environment import Environment


class Agent(ABC):
    def __init__(self, name: str, env: Environment):
        self.name = name
        self.env = env

    @abstractmethod
    def train(self, num_episodes: int) -> List[float]:
        pass
