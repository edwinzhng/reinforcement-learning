import argparse

from agents.dqn import DQN
from environments.environment import Environment

if __name__ == "__main__":
    # Parse arugments from command line
    parser = argparse.ArgumentParser(description="Reinforcement learning algorithm implementations")
    parser.add_argument("-n", "--network", type=str, choices=['DQN', 'A2C', 'PPO'],
                        help="The network to be used", required=True)
    parser.add_argument("-e", "--env", type=str,
                        help="The environment name", required=True)
    parser.add_argument("-p", "--preprocess", type=str, choices=['atari', 'none'],
                        help="Preprocessing function for observations", required=False, default='none')
    parser.add_argument("-f", "--frame-skip", type=int, help="Number of frame skips", required=False, default=1)
    parser.add_argument("-r", "--render", type=bool, help="Render game on screen", required=False, default=False)
    args = parser.parse_args()

    # Build environment
    env = Environment(args.env, args.preprocess, args.render, args.frame_skip)

    # Build model
    model = None
    if args.network == 'DQN':
        model = DQN(env)
    elif args.network == 'A2C':
        model = None
    elif args.network == 'PPO':
        model = None

    # Train model
    model.train()
