# Reinforcement Learning

Implementations of deep reinforcement learning algorithms using Tensorflow 2.0 and OpenAI Gym.

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

```
python src/main.py --agent NETWORK_NAME --env ENV_NAME
```

- Available agents: `DQN`, `A2C`, `PPO`
- Supported environments: `CartPole-v1`, `Acrobot-v1`, `MountainCar-v0`
- Render the OpenAI Gym environment with `--render`
- Enable GPUs for training with `--gpu`
- Normalize environment observations with `--normalization`
- Set number of training episodes with `--episodes`


## References & Code Used

- [Overview of actor-critic methods and A2C](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)
- [PyTorch DQN tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [OpenAI Baselines for implementation reference](https://github.com/openai/baselines)
- [Tensorflow 2 implementations of reinforcement learning algorithms](https://github.com/marload/deep-rl-tf2)

## Algorithms Implemented

- Deep Q Network (DQN) - ["Playing Atari with Deep Reinforcement Learning" (Mnih, 2015)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) and ["Human-level Control Through Deep Reinforcement Learning" (Mnih, 2015)](https://www.nature.com/articles/nature14236)
- Advantage Actor Critic (A2C) - ["Asynchronous Methods for Deep Reinforcement Learning" (Mnih, 2016)](https://arxiv.org/abs/1602.01783)
- Proximal Policy Optimization (PPO) - ["Proximal Policy Optimization Algorithms" (Schulman, 2017)](https://arxiv.org/abs/1707.06347)
