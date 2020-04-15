import gym


class Environment:
    def __init__(self,
                 env_name: str,
                 render: bool):
        self.env = gym.make(env_name)
        self.should_render = render
        self.state_size = len(self.env.observation_space.sample())
        self.action_space_size = self.env.action_space.n

    # Reset environment and return observation
    def reset(self):
        observation = self.env.reset()
        self.render()
        return observation

    # Take one step in the environment based on given action
    def step(self, action: int = None):
        if action is None:
            action = self.random_action()

        observation, reward, terminal, info = self.env.step(action)
        self.render()
        return observation, reward, terminal, info

    # Randomly sample action from environment
    def random_action(self):
        return self.env.action_space.sample()

    # Helper function to render display if the setting is turned on
    def render(self) -> None:
        if self.should_render:
            self.env.render()
