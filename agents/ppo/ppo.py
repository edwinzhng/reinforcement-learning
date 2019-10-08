import tensorflow as tf

from agents.agent import Agent


class PPO(Agent):
    def __init__(self):
        super().__init__('PPO')
