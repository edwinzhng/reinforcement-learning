import argparse
from agents.dqn import dqn
from agents.vpg import vpg
from agents.ppo import ppo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reinforcement Learning Algorithms")

    parser.add_argument("-n", "--network", type=str, choices=['DQN', 'VPG', 'PPO'],
                        help="The network to be used", required=True)
    parser.add_argument("-e", "--env-name", type=str,
                        help="The environment name", required=True)
    parser.add_argument("-p", "--preprocess", type=str, choices=['atari', 'toy'],
                        help="Preprocess function to be used", required=False)
    parser.add_argument("-f", "--frame-skip", type=str, help="Number of frame skips", required=False, default=4)
    parser.add_argument("-l", "--load", type=str, help="The file to load weights from", required=False)
    parser.add_argument("-s", "--save", type=str, help="Folder to save network weights", required=False)
    parser.add_argument("-x", "--stats", type=bool, help="Calculate statistics", required=False)
    parser.add_argument("-d", "--display", type=bool, help="Display game on screen", required=False)

    args = parser.parse_args()

    model = None
    if args.network == 'DQN':
        model = dqn.DQN(args.env_name, args.display, args.preprocess, frame_skip=args.frame_skip)
    elif args.network == 'VPG':
        model = vpg.VPG()
    elif args.network == 'PPO':
        model = ppo.PPO()

    model.train()
