from abc import ABC, abstractmethod

import tensorflow as tf

from environments.environment import Environment


class Agent(ABC):
    def __init__(self,
                 name: str,
                 num_iterations: int,
                 env_name: str,
                 has_display: bool,
                 is_atari: bool,
                 frame_skip: int):
        self.name = name
        self.num_iterations = num_iterations
        self.env = Environment(env_name, has_display, is_atari, frame_skip)

    @abstractmethod
    def train(self):
        pass
