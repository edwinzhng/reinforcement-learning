import gym
import numpy as np

import cv2


class Environment:
    def __init__(self,
                 env_name: str,
                 preprocess: str,
                 render: bool=False,
                 frame_skip=4):
        self.env = gym.make(env_name)
        self.frame_skip = frame_skip
        self.render = render
        self.action_space_size = self.env.action_space.n
        self.preprocess = self.preprocess_default
        if preprocess == 'atari':
            self.preprocess = self.preprocess_atari

    # Reset environment return observation
    def reset(self):
        observation = self.env.reset()
        self.render()
        observation = self.preprocess(observation)
        return self.stack_frames([observation] * self.frame_skip)

    # Take one step in the environment based on given action
    def step(self, action: int = None):
        if action is None:
            action = self.random_action()

        obs = []
        for _ in range(self.frame_skip):
            observation, reward, terminal, info = self.env.step(action)
            obs.append(self.preprocess(observation))
            self.render()

        state = self.stack_frames(obs)
        return state, reward, terminal, info

    # Randomly sample action from environment
    def random_action(self):
        return self.env.action_space.sample()

    # Default preprocessing without any image modifications
    def preprocess_default(self, observation):
        return observation

    # Preprocess Atari images by resizing and converting to grayscale
    def preprocess_atari(self, observation):
        downsampled = cv2.resize(observation, (84,110))
        grayscale = cv2.cvtColor(downsampled, cv2.COLOR_BGR2GRAY)
        observation = grayscale[26:110,:]
        return np.reshape(observation, (1, 84, 84))

    # Stack multiple frames if skip_frames is set
    def stack_frames(self, images):
        return np.stack(images, axis=3).astype('float32')

    # Helper function to render display if the setting is turned on
    def render(self) -> None:
        if self.render:
            self.env.render()
