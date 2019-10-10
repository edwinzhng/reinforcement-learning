from abc import ABC, abstractmethod

import gym
import numpy as np
from gym.wrappers import AtariPreprocessing

import cv2


class Environment():
    def __init__(self,
                 env_name: str,
                 preprocess: str,
                 render_display: bool=False,
                 frame_skip=4):
        self.env = gym.make(env_name)
        self.frame_skip = frame_skip
        self.render_display = render_display
        self.action_space_size = self.env.action_space.n
        self.preprocess = self.preprocess_default
        if preprocess == 'atari':
            self.preprocess = self.preprocess_atari

    def display(self) -> None:
        if self.render_display:
            self.env.render()

    def reset(self) -> object:
        observation = self.env.reset()
        self.display()
        observation = self.preprocess(observation)
        return self.stack_frames([observation] * self.frame_skip)

    def step(self, action: int = None):
        if action is None:
            action = self.random_action()

        obs = []
        for _ in range(self.frame_skip):
            observation, reward, terminal, info = self.env.step(action)
            self.display()
            obs.append(self.preprocess(observation))

        state = self.stack_frames(obs)
        return state, reward, terminal, info

    def random_action(self):
        return self.env.action_space.sample()

    def preprocess_default(self, observation):
        return observation

    def preprocess_atari(self, observation):
        downsampled = cv2.resize(observation, (84,110))
        grayscale = cv2.cvtColor(downsampled, cv2.COLOR_BGR2GRAY)
        observation = grayscale[26:110,:]
        return np.reshape(observation, (1, 84, 84))

    def stack_frames(self, images):
        return np.stack(images, axis=3).astype('float32')