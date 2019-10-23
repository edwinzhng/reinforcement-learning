import numpy as np
import tensorflow as tf

from agents.agent import Agent
from agents.dqn.cnn import ConvolutionalNetwork
from agents.dqn.replay_memory import ReplayMemory


class DQN(Agent):
    def __init__(self,
                 env_name: str,
                 render_display: bool,
                 preprocess: str,
                 num_iterations: int=10e8,
                 memory_size: int=10e7,
                 frame_skip: int=4,
                 batch_size: int=32,
                 gamma: float=0.99,
                 learning_rate: float=0.001):
        super().__init__('DQN', num_iterations, env_name, preprocess, render_display, frame_skip)
        self.replay_memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.gamma = tf.constant(gamma)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.Q = ConvolutionalNetwork(self.env.action_space_size, frame_skip)
        self.Q.model.compile(loss='mse', optimizer=self.optimizer)

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

    @tf.function
    def loss(self, reward, target):
        return tf.reduce_mean(tf.square(reward - target))
