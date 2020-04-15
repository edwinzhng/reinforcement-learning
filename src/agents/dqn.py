from collections import deque
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from agents.agent import Agent
from environments.environment import Environment


class ConvolutionalNetwork(tf.keras.Model):
    def __init__(self, state_size: int, action_space_size: int, frame_skip: int):
        super().__init__()
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.frame_skip = frame_skip

    # model from Playing Atari with Deep Reinforcement Learning (Minh, 2015)
    def call(self, inputs):
        input = tf.keras.Input(shape=(state_size, state_size, self.frame_skip), name='input')
        model = tf.layers.Conv2D(filters=16, kernel_size=(8, 8),
                            strides=4, name='conv_1', activation='relu')(input)
        model = tf.layers.Conv2D(filters=32, kernel_size=(4, 4),
                            strides=2, name='conv_2', activation='relu')(model)
        model = tf.layers.Dense(256, activation='relu', name='fc_1')(model)
        output = tf.layers.Dense(self.action_space_size, activation='softmax', name='output')(model)
        return output

class ReplayMemory:
    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.memory = deque()

    def add(self, state, action, reward, next_state, terminal) -> None:
        while len(self.memory) >= self.memory_size:
            self.memory.popleft()
        self.memory.append((state, action, reward, next_state, terminal))

    def sample(self, batch_size: int) -> List[Tuple]:
        indices = np.random.randint(0, len(self.memory), batch_size)
        return [self.memory[i] for i in indices]

class DQN(Agent):
    def __init__(self,
                 env: Environment,
                 num_iterations: int=10e8,
                 memory_size: int=10e7,
                 frame_skip: int=1,
                 batch_size: int=32,
                 gamma: float=0.99,
                 learning_rate: float=0.001):
        super().__init__('DQN', env)
        self.num_iterations = num_iterations
        self.replay_memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.gamma = tf.constant(gamma)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.Q = ConvolutionalNetwork(self.env.state_size,
                                      self.env.action_space_size,
                                      frame_skip)
        self.Q.compile(loss='mse', optimizer=self.optimizer)

    def predict_action(self, state):
        self.Q.predict(state)

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
                    action = self.Q.model.predict(state)

                next_state, reward, terminal, info = self.env.step(action)
                self.replay_memory.add(state, action, reward, next_state, terminal)
                state = next_state

                replay_sample = self.replay_memory.sample(self.batch_size)
                for state, action, reward, next_state, terminal in replay_sample:
                    self.gradient_descent(state, action, reward, next_state, terminal)
                steps += 1

    @tf.function
    def gradient_descent(self, state, action, reward, next_state, terminal):
        tf.dtypes.cast(reward, tf.float32)
        with tf.GradientTape() as tape:
            target = self.Q.model(state)
            if not terminal:
                next_action = self.Q.model(next_state)
                reward += self.gamma * tf.cast(tf.argmax(next_action), tf.float32)
            loss_value = self.loss(reward, target)

        grads = tape.gradient(loss_value, self.Q.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.Q.model.trainable_variables))
        return loss_value

    @tf.function
    def loss(self, reward, target):
        return tf.reduce_mean(tf.square(reward - target))
