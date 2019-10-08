import tensorflow as tf

from agents.agent import Agent


class VPG(Agent):
    def __init__(self):
        super().__init__('VPG')
