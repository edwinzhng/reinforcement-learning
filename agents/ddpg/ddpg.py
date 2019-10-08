import tensorflow as tf

from agents.agent import Agent


class DDPG(Agent):
    def __init__(self):
        super().__init__('DDPG')
