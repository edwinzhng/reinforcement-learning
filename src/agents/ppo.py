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

class PPO(Agent):
    def __init__(self, env, config):
        super().__init__(self, env)
        self.discount = config['discount']
        self.lmbda = config['lmbda']
        self.clip_ratio = config['clip_ratio']
        self.epochs = config['epochs']
        self.update_steps = config['update_steps']

        self.actor = ActorModel(self.env.state_size, self.env.action_space_size,
                                config['num_layers'], config['hidden_units'])
        self.critic = CriticModel(self.env.state_size, config['num_layers'], config['hidden_units'])
        self.optimizer = tf.keras.optimizers.Adam(lr=config['lr'])

    # Predict action using categorical probability distribution based on policy
    def predict_policy_action(self, state):
        policy = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
        action = tf.squeeze(tf.random.categorical(policy, 1), axis=-1)
        return policy, np.array(action)[0]

    # Compute generalized advantage estimation and target values
    def advantage(self, rewards, values, next_values, done):
        gae = np.zeros_like(rewards)
        targets = np.zeros_like(rewards)

        gae_sum = 0
        forward_value = 0 if done else next_values
        for t in reversed(range(len(rewards))):
            # Compute delta and GAE
            delta = rewards[t] + self.discount * forward_value - values[t]
            gae[t] = self.discount * self.lmbda * gae_sum + delta
            gae_sum = gae_sum

            # Compute targets from GAE and value
            targets[t] = gae[t] + values[t]
            forward_value = values[t]

        return gae, targets

    def train(self, num_episodes):
        episode_rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            states = []
            actions = []
            rewards = []
            old_policies = []
            while not done:
                policy, action = self.predict_policy_action(state)
                next_state, reward, done, _ = self.env.step(action)

                states.append(np.expand_dims(state, 0))
                actions.append(np.expand_dims(action, 0))
                # Scale reward to optimize frequency over value
                rewards.append(np.expand_dims(reward * 0.01, 0))
                old_policies.append(policy)

                total_reward += reward
                state = next_state

                if len(states) >= self.update_steps or done:
                    states = np.vstack(states)
                    actions = np.vstack(actions)
                    rewards = np.vstack(rewards)
                    old_policies = np.vstack(old_policies)

                    # Compute values and advantages for loss
                    values = self.critic(tf.convert_to_tensor(states))
                    next_values = self.critic(tf.convert_to_tensor([next_state]))
                    gae, targets = self.advantage(rewards, values, next_values, done)
                    for _ in range(self.epochs):
                        self.update_actor(old_policies, states, actions, gae)
                        self.update_critic(states, targets)

                    states = []
                    actions = []
                    rewards = []
                    old_policies = []

            episode_rewards.append(total_reward)
            print(f'Episode: {episode + 1} Reward: {total_reward}')
        return episode_rewards

    # Update actor weights based on loss
    def update_actor(self, old_policies, states, actions, gae):
        old_policies = tf.stop_gradient(old_policies)
        gae = tf.stop_gradient(gae)

        # Convert actions to one-hot vectors to calculate policy difference
        actions = tf.cast(tf.one_hot(actions, self.env.action_space_size), dtype=tf.float64)
        actions = tf.reshape(actions, [-1, self.env.action_space_size])
        with tf.GradientTape() as tape:
            policy = self.actor(states)
            loss = self.actor_loss(old_policies, policy, actions, gae)

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

    # Actor loss based on difference between new and old policies
    def actor_loss(self, old_policy, new_policy, actions, gae):
        log_old_policy = tf.math.log(tf.reduce_sum(old_policy * actions))
        log_new_policy = tf.math.log(tf.reduce_sum(new_policy * actions))

        # Compute ratio between policies and clip
        ratio = tf.math.exp(log_new_policy - log_old_policy)
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

        # Total surrogate loss
        surrogate_loss = -tf.minimum(ratio * gae, clipped_ratio * gae)
        return tf.reduce_mean(surrogate_loss)

    # Mean squared error for critic value loss
    def critic_loss(self, values, targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(values, targets)
