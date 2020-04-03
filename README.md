# Reinforcement Learning

Implementations of deep reinforcement learning algorithms using PyTorch and OpenAI Gym.

## Setup

Install [Anaconda](https://www.anaconda.com/distribution) for Python 3.7

Create an Anaconda environment
 ```
 conda create -n rl
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

```
python main.py \
    --network NETWORK_NAME \
    --env-name ENV_NAME
```

## Algorithms Implemented

* Deep Q Network (DQN) - ["Playing Atari with Deep Reinforcement Learning" (Mnih, 2015)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* Advantage Actor Critic (A2C) - ["Asynchronous Methods for Deep Reinforcement Learning" (Mnih, 2016)](https://arxiv.org/abs/1602.01783)
* Proximal Policy Optimization (PPO) - ["Proximal Policy Optimization Algorithms" (Schulman, 2017)](https://arxiv.org/abs/1707.06347)
