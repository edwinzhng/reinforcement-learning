import random
from collections import deque

import numpy as np
import tensorflow as tf

from agents.agent import Agent


# Function approximator for the Q value in DQN
class QModel(tf.keras.Model):
    def __init__(self,
                 state_size,
                 action_space_size,
                 num_layers,
                 hidden_units):
        super(QModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer((state_size,))
        self.hidden_layers = []
        for _ in range(num_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_units, activation='relu'))
        self.output_layer = tf.keras.layers.Dense(action_space_size)

    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output

class DQN(Agent):
    def __init__(self, env, config):
        super().__init__('DQN', env)
        self.discount = config['discount']
        self.batch_size = config['batch_size']
        self.target_update_steps = config['target_update_steps']
        self.num_episodes = config['num_episodes']
        self.epsilon = config['epsilon']
        self.epsilon_decay = config['epsilon_decay']
        self.replay_memory = deque(maxlen=config['memory_size'])

        # Initialize main and target model
        self.Q = QModel(self.env.state_size, self.env.action_space_size,
                        config['num_layers'], config['hidden_units'])
        self.Q_target = QModel(self.env.state_size, self.env.action_space_size,
                        config['num_layers'], config['hidden_units'])
        self.optimizer = tf.keras.optimizers.Adam(lr=config['lr'])

    # Predict action with random probability of epsilon
    def predict_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return self.env.random_action()
        else:
            q = self.Q(tf.convert_to_tensor([state], dtype=tf.float32))[0]
            return np.argmax(q)

    def train(self):
        step = 0
        for episode in range(1, self.num_episodes + 1):
            # Reset environment for new episode
            state = self.env.reset()
            total_reward = 0
            done = False
            epsilon = self.epsilon * (self.epsilon_decay ** episode)

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
