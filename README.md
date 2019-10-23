# Reinforcement Learning

Implementations of deep reinforcement learning algorithms using TensorFlow and Gym.

## Setup

1. Install system dependencies
    - MacOS `brew install cmake boost boost-python sdl2 swig wget`
    - Ubuntu `apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig`
2. `virtualenv /env`
3. `source /env/bin/activate/`
4. `pip install -r requirements.txt`

## Running An Agent

```
python main.py \
    --network NETWORK_NAME \
    --env-name ENV_NAME \
    --preprocessing PREPROCESSING_FUNCTION
```

## Algorithms Implemented

* Deep Q Network (DQN) - ["Playing Atari with Deep Reinforcement Learning" (Minh, 2015)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* Vanilla Policy Gradient (VPG)

## In Progress / Planned

* Proximal Policy Optimization (PPO)
* Deep Deterministic Policy Gradients (DDPG)
* Trust Region Policy Optimization (TRPO)
* Advantage Actor Critic (A2C)
* Asynchronous Advantage Actor Critic (A3C)
* Generative Adversarial Imitation Learning (GAIL)
* Parallelized DQN and PPO

