from abc import ABC, abstractmethod

from environments.environment import Environment


class Agent(ABC):
    def __init__(self, name: str, env: Environment):
        self.name = name
        self.env = env

    @abstractmethod
    def predict_action(self, state):
        pass

    @abstractmethod
    def train(self):
        pass

    # Evaluate agent on environment and return reward value
    def evaluate(self):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = self.predict_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            self.env.render()

        return total_reward
