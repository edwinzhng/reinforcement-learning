import argparse
import csv
import os

import matplotlib.pyplot as plt
import pandas as pd
import pkg_resources


# Compute exponential moving average
def average(rewards, weight = 0.98):
    last = rewards[0]
    averages = []
    for reward in rewards:
        avg = last * weight + (1 - weight) * reward
        averages.append(avg)
        last = avg
    return averages

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', type=str, required=True)
    parser.add_argument('--episodes', type=int, default = 1000, required=False)
    args = parser.parse_args()

    agents = ['DQN', 'A2C', 'PPO']
    colors = ['r', 'b', 'g']
    for agent, color in zip(agents, colors):
        path = pkg_resources.resource_filename(
            __name__, f'logs/{agent}_{args.env}.csv')
        df = pd.read_csv(path, names=['episode', 'reward'], skipinitialspace=True)
        data = [df['episode'].values[1:].astype(int), df['reward'].values[1:].astype(float)]

        # Plot data and average reward
        plt.plot(data[0], data[1], alpha=0.1, color=color)
        plt.plot(data[0], average(data[1]), color=color, label=agent)

    plt.xlim(0, args.episodes)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(args.env)
    plt.legend(loc='upper left')
    plt.show()
