import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, optimizers

from agents.agent import Agent


# Function approximator for the Q value in DQN
class QModel(tf.keras.Model):
    def __init__(self,
                 state_size,
                 action_space_size,
                 num_layers = 3,
                 hidden_units = 128):
        super().__init__()
        self.input_layer = tf.keras.layers.InputLayer((state_size,), name='input')
        self.hidden_layers = []
        for i in range(num_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(
                hidden_units, activation='relu', name=f'hidden_layer_{i}'))
        self.output_layer = tf.keras.layers.Dense(action_space_size, name='output')

    @tf.function
    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output

class DQN(Agent):
    def __init__(self,
                 env,
                 lr=0.001,
                 discount=0.99,
                 batch_size=64,
                 target_update_steps=25,
                 num_episodes=3000,
                 memory_size=2000):
        super().__init__('DQN', env)
        self.discount = discount
        self.batch_size = batch_size
        self.target_update_steps = target_update_steps
        self.num_episodes = num_episodes
        self.replay_memory = deque(maxlen=memory_size)

        # Initialize main and target model
        self.Q = QModel(self.env.state_size, self.env.action_space_size)
        self.Q_target = QModel(self.env.state_size, self.env.action_space_size)
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

    # Predict action with random probability of epsilon
    def predict_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return self.env.random_action()
        else:
            q = self.Q(tf.convert_to_tensor([state], dtype=tf.float32))[0]
            return np.argmax(q)

    def train(self):
        episode = 0
        step = 0

        while episode < self.num_episodes:
            # Reset environment for new episode
            state = self.env.reset()
            total_reward = 0
            done = False
            epsilon = 1 / (episode * 0.1 + 1)
            episode += 1

            while not done:
                # Take action based on current state and append to replay memory
                action = self.predict_action(state, epsilon)
                next_state, reward, done, info = self.env.step(action)
                self.replay_memory.append((state, action, reward, next_state, done))

                # Update variables
                step += 1
                state = next_state
                total_reward += reward

                # Continue if not enough samples yet
                if len(self.replay_memory) < self.batch_size:
                    continue

                # Update network weights and target network
                self.update_weights(step)

            print(f'Episode: {episode} Reward: {total_reward}')

    def update_weights(self, step):
        # Sample batch from replay memory
        replay_samples = random.sample(self.replay_memory, self.batch_size)
        states = tf.convert_to_tensor([x[0] for x in replay_samples], dtype=tf.float32)
        actions = tf.convert_to_tensor([x[1] for x in replay_samples], dtype=tf.int32)
        rewards = tf.convert_to_tensor([x[2] for x in replay_samples], dtype=tf.float32)
        next_states = tf.convert_to_tensor([x[3] for x in replay_samples], dtype=tf.float32)
        dones = tf.convert_to_tensor([x[4] for x in replay_samples], dtype=tf.float32)

        # Compute gradients
        with tf.GradientTape() as tape:
            # Calculate y as action-value estimate
            y_q = self.Q_target(tf.concat(next_states, axis=0))
            next_action = tf.argmax(y_q, axis=1)
            y = tf.reduce_sum(tf.one_hot(next_action, self.env.action_space_size) * y_q, axis=1)
            y = rewards + (1 - dones) * self.discount * y

            # Calculate target value for loss
            target_q = self.Q(tf.concat(states, axis=0))
            target = tf.reduce_sum(tf.one_hot(actions, self.env.action_space_size) * target_q, axis=1)

            loss = tf.reduce_mean(tf.square(y - target))

        grads = tape.gradient(loss, self.Q.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.Q.trainable_variables))

        # Copy weights from main network to target network
        if step % self.target_update_steps == 0:
            self.Q_target.set_weights(self.Q.get_weights())