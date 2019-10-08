import tensorflow as tf

from agents.agent import Agent


class TRPO(Agent):
    def __init__(self):
        super().__init__('TRPO')
