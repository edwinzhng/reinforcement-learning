import numpy as np
import tensorflow as tf

from agents.agent import Agent


class ActorModel(tf.keras.Model):
    def __init__(self, state_size, action_space_size, num_layers, hidden_units):
        super(ActorModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer((state_size,))
        self.hidden_layers = []
        for _ in range(num_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_units, activation='relu'))
        self.policy = tf.keras.layers.Dense(action_space_size, activation='softmax')

    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        policy = self.policy(x)
        return policy

class CriticModel(tf.keras.Model):
    def __init__(self, state_size, num_layers, hidden_units):
        super(CriticModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer((state_size,))
        self.hidden_layers = []
        for _ in range(num_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_units, activation='relu'))
        self.value = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        value = self.value(x)
        return value

class A2C(Agent):
    def __init__(self, env, config):
        super().__init__(self, env)
        self.discount = config['discount']
        self.entropy = config['entropy']
        self.update_steps = config['update_steps']
        self.num_episodes = config['num_episodes']

        self.actor = ActorModel(self.env.state_size, self.env.action_space_size,
                                config['num_layers'], config['hidden_units'])
        self.critic = CriticModel(self.env.state_size, config['num_layers'], config['hidden_units'])
        self.optimizer = tf.keras.optimizers.Adam(lr=config['lr'])

    # Predict action using categorical probability distribution based on policy
    def predict_action(self, state):
        policy = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
        action = tf.squeeze(tf.random.categorical(policy, 1), axis=-1)
        return np.array(action)[0]

    # Compute advantage and target values
    def advantage(self, reward, next_state, done, value):
        if done:
            target = reward
        else:
            next_value = self.critic(tf.convert_to_tensor([next_state]))
            target = reward + self.discount * next_value

        advantage = target - value
        return advantage, target

    def train(self):
        for episode in range(1, self.num_episodes + 1):
            state = self.env.reset()
            total_reward = 0
            done = False

            states = []
            actions = []
            advantages = []
            targets= []
            while not done:
                action = self.predict_action(state)
                next_state, reward, done, info = self.env.step(action)

                value = self.critic(tf.convert_to_tensor([state]))
                advantage, target = self.advantage(reward * 0.01, next_state, done, value)

                states.append(np.expand_dims(state, 0))
                actions.append(np.expand_dims(action, 0))
                advantages.append(advantage)
                targets.append(target)

                total_reward += reward
                state = next_state

                if len(states) >= self.update_steps or done:
                    states = np.vstack(states)
                    actions = np.vstack(actions)
                    advantages = np.vstack(advantages)
                    targets = np.vstack(targets)

                    self.update_actor(states, actions, advantages)
                    self.update_critic(states, targets)

                    states = []
                    actions = []
                    advantages = []
                    targets = []

            print(f'Episode: {episode} Reward: {total_reward}')

    # Update actor weights based on loss
    def update_actor(self, states, actions, advantages):
        tf.stop_gradient(advantages)
        actions = tf.cast(actions, tf.int32)

        with tf.GradientTape() as tape:
            policy = self.actor(states)
            loss = self.actor_loss(policy, actions, advantages)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

    # Update critic weights based on loss
    def update_critic(self, states, targets):
        targets = tf.stop_gradient(targets)
        with tf.GradientTape() as tape:
            values = self.critic(states)
            loss = self.critic_loss(values, targets)

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

    # Actor loss based on sparse categorical crossentropy
    def actor_loss(self, policy, actions, advantages):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = ce_loss(actions, policy, sample_weight=advantages)

        # Add entropy
        entropy_term = tf.keras.losses.categorical_crossentropy(policy, policy, from_logits=False)
        loss -= self.entropy * entropy_term
        return loss

    # Mean squared error for critic value loss
    def critic_loss(self, values, targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(values, targets)
