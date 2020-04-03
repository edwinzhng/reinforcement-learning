from abc import ABC, abstractmethod

from environments.environment import Environment


class Agent(ABC):
    def __init__(self, name: str, env: Environment):
        self.name = name
        self.env = env

    @abstractmethod
    def train(self):
        pass
