from abc import ABC, abstractmethod

import gym
from gym.wrappers import AtariPreprocessing


class Environment():
    def __init__(self,
                 env_name: str,
                 has_display: bool=False,
                 is_atari: bool=False,
                 frame_skip=4):
        self.env = gym.make(env_name)
        self.frame_skip = frame_skip
        self.has_display = has_display
        self.action_space_size = self.env.action_space.n
        if is_atari:
            self.env = AtariPreprocessing(self.env, frame_skip=self.frame_skip)

    def display(self) -> None:
        if self.has_display:
            self.env.render()

    def reset(self) -> object:
        observation = self.env.reset()
        self.display()
        return observation

    def step(self, action: int = None):
        if action is None:
            action = self.random_action()
        observation, reward, terminal, info = self.env.step(action)
        self.display()
        return observation, reward, terminal, info

    def random_action(self):
        return self.env.action_space.sample()