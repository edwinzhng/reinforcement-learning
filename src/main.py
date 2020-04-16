import argparse

from agents.a2c import A2C
from agents.dqn import DQN
from environments.environment import Environment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reinforcement learning algorithm implementations")
    parser.add_argument("-a", "--agent", type=str, choices=['DQN', 'DDQN', 'A2C', 'PPO'],
                        help="The agent to be used", required=True)
    parser.add_argument("-e", "--env", type=str,
                        help="The environment name", required=True)
    parser.add_argument("-r", "--render", type=bool, help="Render game on screen", required=False, default=False)
    args = parser.parse_args()

    # Build environment
    env = Environment(args.env, args.render)

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
