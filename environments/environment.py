from abc import ABC, abstractmethod

import gym


class Environment(ABC):
    def __init__(self, env_name: str, has_display: bool):
        self.env = gym.make(env_name)
        self.has_display = has_display

    def display(self) -> None:
        if has_display:
            self.env.render()

    def reset(self) -> object:
        observation = self.env.reset()
        self.display()
        return observation

    def step(self, action: int = None):
        # random action
        if action is None:
            action = env.action_space.sample()
        observation, reward, terminal, info = env.step(action)
        self.display()

    @abstractmethod
    def preprocess(self):
        pass
