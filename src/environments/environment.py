import gym
import numpy as np

from sklearn.preprocessing import StandardScaler


class Environment:
    def __init__(self,
                 env_name: str,
                 render: bool,
                 normalize: bool):
        self.env = gym.make(env_name)
        self.should_render = render
        self.normalize = normalize
        self.state_size = len(self.env.observation_space.sample())

        self.continuous = False
        self.action_space_size = None
        self.action_space_low = None
        self.action_space_high = None

        if type(self.env.action_space) == gym.spaces.Discrete:
            self.action_space_size = self.env.action_space.n
        else:
            self.continuous = True
            self.action_space_low = self.env.action_space.low[0]
            self.action_space_high = self.env.action_space.high[0]

        # Sample observation space to scale inputs
        observation_samples = [self.env.observation_space.sample() for _ in range(10000)]
        self.scaler = StandardScaler()
        self.scaler.fit(observation_samples)

    # Reset environment and return observation
    def reset(self):
        state = self.env.reset()
        self.render()
        return self.normalize_state(state)

    # Take one step in the environment based on given action
    def step(self, action: int = None):
        if action is None:
            action = self.random_action()

        state, reward, done, info = self.env.step(action)
        self.render()
        return self.normalize_state(state), reward, done, info

    # Randomly sample action from environment
    def random_action(self):
        return self.env.action_space.sample()

    # Helper function to render display if the setting is turned on
    def render(self) -> None:
        if self.should_render:
            self.env.render()

    def normalize_state(self, state):
        if self.normalize:
            return self.scaler.transform([state])[0]
        return state
