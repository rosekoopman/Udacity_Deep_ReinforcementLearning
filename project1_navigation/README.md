# README

This package contains a deep reinforcement learning agent to solve the "Banana navigation" problem

## Installation

Clone this repository
```bash
git clone https://github.com/rosekoopman/navigation_DQN.git
```

Download the Banana navigation app and place it in the root folder of this repository. Pick the download link according to your OS: 


- Linux: [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Note that the Banana environment provided here is slightly different from the Banana environment by unity agents on [github](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector) . Due to this fact, we will need to install an **old** version of the python package unity-agents: `unityagents==0.4.0`. In order to be able to install this old version, we need an **old** python version, for instance `python 3.6`. Once you have your python setup you can install the required packages using the requirements.txt . It is recommended to run the installation in a separate virtual environment.

```bash
pip install -r requirements.txt
```


## Environment

The agent lives in a square world and aims to collect as many yellow bananas, avoiding all blue bananas. The reward of collecting a yellow banana is `+1` while the reward for collecting a blue banana is `-1`. The task is episodic. The action space contains 4 discrete actions

- 0: forward
- 1: backward
- 2: left
- 3: right

The state space has a dimension of 37 and all dimensions are continuous. The dimensions relate to the velocity of the agent and a ray based perception of the environment.

The problem is considered solved if the agent achieves a average total score of at least `+13` in 100 subsequent episodes.

## Usage

### Train the agent

The easiest way to train the agent is to use the notebook `navigation.ipynb`. Hit `run all` to train the agent using DQN *and* DDQN. 


### Use the agent

```python
from unityagents import UnityEnvironment
import numpy as np
from agents import Agent

# load the Banana app
env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")

# get app info
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
action_size = brain.vector_action_space_size
state_size = brain.vector_observation_space_size

# instatiate the DRL agent
agent = Agent(state_size=state_size, action_size=action_size, seed=0) 

# load network weights
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
```

