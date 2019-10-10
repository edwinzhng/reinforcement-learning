import numpy as np
import tensorflow as tf

from agents.agent import Agent
from agents.dqn.cnn import ConvolutionalNetwork
from agents.dqn.replay_memory import ReplayMemory


class DQN(Agent):
    def __init__(self,
                 env_name: str,
                 has_display: bool=False,
                 is_atari: bool=False,
                 num_iterations: int=10e8,
                 memory_size: int=10e7,
                 frame_skip: int=4,
                 batch_size: int=32,
                 gamma: float=0.99):
        super().__init__('DQN', num_iterations, env_name, has_display, is_atari, frame_skip)
        self.replay_memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.Q = ConvolutionalNetwork(self.env.action_space_size)

    def train(self):
        steps = 0

        while steps < self.num_iterations:
            observation = self.env.reset()
            epsilon = max(0.1, 1 - steps / 1e7)

            terminal = False
            while not terminal:
                if np.random.random() < epsilon:
                    action = self.env.random_action()
                else:
                    action = self.Q.model.predict(observation)

                previous_obs = observation
                observation, reward, terminal, info = self.env.step(action)

                self.replay_memory.add(observation, action, reward, terminal)
                self.replay_sample = self.replay_memory.sample(self.batch_size)

                self.gradient_descent()
                steps += 1


    def gradient_descent(self):
        pass

if __name__=="__main__":
    dqn = DQN('Breakout-v4', True, True)
    dqn.train()
