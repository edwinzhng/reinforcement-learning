from abc import ABC, abstractmethod

import tensorflow as tf


class Agent(ABC):
    def __init__(self, name: str, num_iterations: int):
        self.name = name
        self.num_iterations = num_iterations

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
