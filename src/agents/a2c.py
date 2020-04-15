import numpy as np
import tensorflow as tf

from agents.agent import Agent
from environments.environment import Environment


class ProbabilityDistModel(tf.keras.Model):
  def call(self, logits, **kwargs):
    return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(tf.keras.Model):
  def __init__(self, num_actions):
    pass

  def call(self, inputs, **kwargs):
    pass

  def action_value(self, obs):
    pass

class A2C(Agent):
    def __init__(self,
                 env: Environment,
                 num_iterations: int=10e8,
                 memory_size: int=10e7,
                 batch_size: int=32,
                 gamma: float=0.99,
                 learning_rate: float=0.001):
        super().__init__('A2C', env)
        self.num_iterations = num_iterations

    def predict_action(self, state):
        pass

    def train(self):
        steps = 0
        while steps < self.num_iterations:
            state = self.env.reset()
            epsilon = max(0.1, 1 - steps / 1e7)

            terminal = False
            while not terminal:
                if np.random.random() < epsilon:
                    action = self.env.random_action()
                else:
                    action = self.predict_action(state)

                next_state, reward, terminal, info = self.env.step(action)
                self.replay_memory.add(state, action, reward, next_state, terminal)
                state = next_state

                replay_sample = self.replay_memory.sample(self.batch_size)
                for state, action, reward, next_state, terminal in replay_sample:
                    self.gradient_descent(state, action, reward, next_state, terminal)
                steps += 1

    @tf.function
    def gradient_descent(self, state, action, reward, next_state, terminal):
        pass

    @tf.function
    def loss(self, reward, target):
        pass
