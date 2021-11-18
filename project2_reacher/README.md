# README

This package contains a deep reinforcement learning agent to solve the "Continuous Control" problem where a double jointed arm follows a target object.

## Installation

Clone this repository

    git clone https://github.com/rosekoopman/continuous_control_DDPG.git
    
Download the Continuous control app, uncompress it, and place it in the root folder of this repository. Pick the download link according to your OS:

- Linux [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows 32bit [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows 64bit [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Install the required python packages using the `requirements.txt`:

    pip install -r requirements.txt
    
## Environment

The *Reacher* agent is a double-jointed arm which can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

A 20-dimensional *Reacher* environment is used, consisting of 20 agents with double-jointed arms. These 20 agents share one *memory*, which enhences training. 

The agent is considered sufficiently *trained* when the achieved average score over 100 subsequent episodes is at least 30, averaged over all 20 agents.

## Usage

### Train the agent

The easiest way to train the agent is to use the notebook `Continuous_Control_20agents.ipynb`. Different hyperparameter options are provided. Option 5, using a larger replay buffer, shows slightly better performance than the other options. Option 0, the baseline, needs the least number of episodes for training although differences in training time are marginal.


### Use the agent

    from unityagents import UnityEnvironment
    import torch
    from agent import Agent

    # load the Reacher app
    env = UnityEnvironment(file_name='/data/Reacher_Linux_NoVis/Reacher.x86_64')
    
    # get app info
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size

    # instantiate the DDPG agent
    agent = Agent(num_agents, state_size, action_size, 4, 1, 1)

    # load network weights to agent
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))