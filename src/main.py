import argparse
import os

from agents.a2c import A2C
from agents.dqn import DQN
from environments.environment import Environment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reinforcement learning algorithm implementations')
    parser.add_argument('-a', '--agent', type=str, choices=['DQN', 'A2C', 'PPO'],
                        help='The agent to be used', required=True)
    parser.add_argument('-e', '--env', type=str,
                        help='The environment name', required=True)
    parser.add_argument('-r', '--render', help='Render game on screen', action='store_true')
    parser.add_argument('-g', '--gpu', help='Use GPU for training', action='store_true')
    parser.add_argument('-n', '--normalize', help='Normalize inputs', action='store_true')
    args = parser.parse_args()

    if not args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Build environment
    env = Environment(args.env, args.render, args.normalize)

    # Build model
    model = None
    if args.agent == 'DQN':
        model = DQN(env)
    elif args.agent == 'A2C':
        model = A2C(env)
    elif args.agent == 'PPO':
        model = None

    # Train model
    model.train()
