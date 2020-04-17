import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from agents.agent import Agent


class ContinuousActorModel(tf.keras.Model):
    def __init__(self, state_size, num_layers = 3, hidden_units = 128):
        super().__init__()
        self.input_layer = tf.keras.layers.InputLayer((state_size,), name='input')
        self.hidden_layers = []
        for i in range(num_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(
                hidden_units, activation='relu', name=f'hidden_layer_{i}'))

        # Continuous action space policy heads for mean and standard deviation
        # Parameterized from original A3C paper
        self.mean = tf.keras.layers.Dense(1, activation='relu', name='mean')
        self.stddev = tf.keras.layers.Dense(1, activation='softplus', name='stddev')

    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        mean = self.mean(x)
        stddev = self.stddev(x) + 1e-5
        policy = tf.squeeze(tf.concat([[mean], [stddev]], axis=0))
        return policy

class DiscreteActorModel(tf.keras.Model):
    def __init__(self, state_size, action_space_size, num_layers = 3, hidden_units = 128):
        super().__init__()
        self.input_layer = tf.keras.layers.InputLayer((state_size,), name='input')
        self.hidden_layers = []
        for i in range(num_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(
                hidden_units, activation='relu', name=f'hidden_layer_{i}'))

        # Discrete action space policy
        self.policy = tf.keras.layers.Dense(action_space_size,
                                            activation='softmax',
                                            name='policy')

    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        policy = self.policy(x)
        return policy

class CriticModel(tf.keras.Model):
    def __init__(self,
                 state_size,
                 num_layers = 2,
                 hidden_units = 64):
        super().__init__()
        self.input_layer = tf.keras.layers.InputLayer((state_size,), name='input')
        self.hidden_layers = []
        for i in range(num_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(
                hidden_units, activation='relu', name=f'hidden_layer_{i}'))
        self.critic = tf.keras.layers.Dense(hidden_units, activation='relu')
        self.value = tf.keras.layers.Dense(1, name='value')

    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        critic = self.critic(x)
        value = self.value(critic)
        return value

class A2C(Agent):
    def __init__(self,
                 env,
                 lr=3e-4,
                 discount=0.99,
                 entropy=0.01,
                 max_steps=128,
                 num_episodes=3000):
        super().__init__('A2C', env)
        self.discount = discount
        self.entropy = entropy
        self.max_steps = max_steps
        self.num_episodes = num_episodes

        if self.env.continuous:
            self.actor = ContinuousActorModel(self.env.state_size)
        else:
            self.actor = DiscreteActorModel(self.env.state_size, self.env.action_space_size)
        self.critic = CriticModel(self.env.state_size)
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

    # Predict action using categorical probability distribution based on policy
    def predict_action(self, state):
        policy = self.actor(tf.convert_to_tensor(np.array([state]), dtype=tf.float32))
        if self.env.continuous:
            # Sample from continuous action space if action is continuous
            mu, sigma = np.array(policy)
            gaussian = tfp.distributions.Normal(mu, sigma)
            action = tf.squeeze(gaussian.sample(1), axis=0)
            action = tf.clip_by_value(action + 1e-5,
                                      self.env.action_space_low,
                                      self.env.action_space_high)
            action = np.array([action])
        else:
            # Otherwise choose an action from the discrete space
            action = tf.squeeze(tf.random.categorical(policy, 1), axis=-1)
            action = np.array(action)[0]
        return action

    def train(self):
        episode = 0
        step = 0
        total_reward = 0
        state = self.env.reset()
        done = False

        while episode < self.num_episodes:
            samples = []
            start_step = step

            while step - start_step < self.max_steps:
                action = self.predict_action(state)
                next_state, reward, done, info = self.env.step(action)
                samples.append((state, action, reward, next_state, done))

                state = next_state
                total_reward += reward
                step += 1

                if done:
                    print(f'Episode: {episode} Reward: {total_reward}')
                    episode += 1
                    total_reward = 0
                    state = self.env.reset()
                    break

            self.update_weights(samples)

    def update_weights(self, samples):
        states = [s[0] for s in samples]
        actions = [s[1] for s in samples]
        rewards = [s[2] for s in samples]
        next_states = [s[3] for s in samples]
        dones = [s[4] for s in samples]

        # Update Actor weights with entropy
        with tf.GradientTape() as tape:
            advantages = self.advantages(states, rewards, dones)
            policy = self.actor(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
            tape.watch(policy)
            if self.env.continuous:
                # Negative log loss if action space is continuous
                mu, sigma = policy
                gaussian = tfp.distributions.Normal(mu, sigma)
                loss = tf.math.log(gaussian.prob(actions))

                # Entropy term for continuous action spaces from the A3C paper
                entropy = -0.5 * (tf.math.log(2 * np.pi * tf.math.square(sigma)) + 1)
            else:
                # Sparse categorical crossentropy if action space is discrete
                crossentropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
                loss = crossentropy_loss(actions, policy)

                # Entropy from categorical crossentropy
                entropy = tf.keras.losses.categorical_crossentropy(policy, policy, from_logits=False)

            loss = tf.reduce_mean(loss * advantages)
            actor_loss = loss - (self.entropy * tf.cast(entropy, tf.float32))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Update Critic weights
        with tf.GradientTape() as tape:
            advantages = self.advantages(states, rewards, dones)
            critic_loss = tf.reduce_mean(tf.square(advantages))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    # Compute advantage based on reward values
    def advantages(self, states, rewards, dones):
        # Initialize reward sum based on last state
        last_state = states[-1]
        if dones[-1]:
            R = 0
        else:
            R = self.critic(tf.convert_to_tensor(np.expand_dims(last_state, axis=0), dtype=tf.float32))

        # Compute discounted rewards using discount value and running sum R
        discounted_rewards = []
        for reward in reversed(rewards):
            R = reward + self.discount * R
            discounted_rewards.append(R)
        discounted_rewards.reverse()

        # Compute advantages by subtracting discounted reward from values
        values = self.critic(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
        advantages = discounted_rewards - values
        return advantages
