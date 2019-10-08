import gym

class Environment:
    def __init__(self):
        pass

    def run_environment(self, env_name='CardPole-V0'):
        env = gym.make(env_name)
        env.reset()
        for _ in range(1000):
            env.render()
            env.step(env.action_space.sample())
        env.close()
