# Reinforcement Learning

Implementations of deep reinforcement learning algorithms using PyTorch and OpenAI Gym.

## Setup

Install [Anaconda](https://www.anaconda.com/distribution) for Python 3.7

Create an Anaconda environment
 ```
 conda create -n rl python=3.7
 ```

Activate the Anaconda environment
```
conda activate rl
```

Install Dependencies
```
pip install -r requirements.txt
```

## Running An Agent

Available network names: `DQN`, `A2C`, `PPO`

```
python src/main.py \
    --network NETWORK_NAME \
    --env-name ENV_NAME
```

## Algorithms Implemented

* Deep Q Network (DQN) - ["Playing Atari with Deep Reinforcement Learning" (Mnih, 2015)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) and ["Human-level Control Through Deep Reinforcement Learning" (Mnih, 2015)](https://www.nature.com/articles/nature14236)
* Advantage Actor Critic (A2C) - ["Asynchronous Methods for Deep Reinforcement Learning" (Mnih, 2016)](https://arxiv.org/abs/1602.01783)
* Proximal Policy Optimization (PPO) - ["Proximal Policy Optimization Algorithms" (Schulman, 2017)](https://arxiv.org/abs/1707.06347)
