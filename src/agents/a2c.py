import numpy as np
import tensorflow as tf

from agents.agent import Agent


class A2C(Agent):
    def __init__(self, env):
        super().__init__('A2C', env)

    def train(self):
        pass
