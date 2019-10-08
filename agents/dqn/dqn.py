import tensorflow as tf

from agents.agent import Agent


class DQN(Agent):
    def __init__(self):
        super().__init__('DQN')
