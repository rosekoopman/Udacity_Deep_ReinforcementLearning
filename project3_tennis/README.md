# tennis_drl


This package contains two implementations of deep reinforcement learning agents to solve the "Tennis" problem 
where two agents play a game of tennis.

## Installation

Clone this repository

    git clone https://github.com/rosekoopman/tennis_drl.git

Download the Tennis app, uncompress it, and place it in the root folder of this repository. 
Pick the download link according to your OS. Note that these evironment are slightly 
different from the official Unity Tennis environment.

- Linux [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows 32bit [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows 64bit [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

When working on your local system, you need to follow the installation instructions as outlined [here](https://github.com/udacity/deep-reinforcement-learning#dependencies) to setup your environment correctly. A `requirements.txt` is included in
the tennis_dlr repo which can be used to check/install the dependencies that I used.
    
When working on the Udacity Workspace, you can simply install the required python packages by running

    pip -q install ./python

This command can either be run from the command line or from within the `Tennis.ipynb` notebook.


## Environment

The *Tennis* environment consists of two tennis rackets. The aim is to move the tennis rackets
such that a ball is bounched back and forth over the net, without dropping the ball or
hitting the ball outside of the boundaries of the playing field. Each racket is controlled
by an agent.

The observation space consists of 8 variables corresponding to the position and velocity of 
the ball and racket. Each agent receives its own, local observation. Three subsequent 
local observations are stacked, in order to preserve temporal information on ball and racket 
movement direction. Each agent therefore has an observation space consisting of 24 continuous
elements. The *Tennis* environment provides a new local state for both agents simultaneously, given the
actions of both of the agents. The action space consists of two continuous variables, 
corresponding to the movement towards (and away) from the net and jumping. 

The *Tennis* environment provides the state, reward and done of both agents. Note that as
there are now two agents, there are also two rewards. For training the agent, the reward
corresponding the individual agent is used. However, to determine the total score achieved
in an episode the maximum score of the two agents is used. The reward structure of the
environment is as follows:

- When the agent hits the ball over the net the agent receives +0.1
- When the agent lets the ball hit the ground, or when the agent hits the ball out of the 
  boundaries of the game, it receives -0.01 

The agents are considered sufficiently trained, when the achieved average score over 100 
subsequent episodes is at least 0.5.


## Usage

Two trained agents are included in this repository, using either the MADDPG algorithm suitable
for environments where mulitple agents interact, or the more simple DDPG agent.

### Use the MADDPG agent

The example code below shows how load and use the MADDPG agent. The code is assumed to 
be run from the MADDPG folder in this repo.

    from unityagents import UnityEnvironment
    import torch
    from agents import MADDPG
    import numpy as np

    # load the Tennis app
    env = UnityEnvironment(file_name="/data/Tennis_Linux_NoVis/Tennis")

    # get Tennis app info
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]          # reset the environment 
    states = env_info.vector_observations
    state_size = states.shape[1]                               # get state size
    action_size = brain.vector_action_space_size               # get action size
    q_input_size = int(2 * (state_size + action_size))         # get critic input size
    num_agents = len(env_info.agents)                          # get number of agents

    # instantiate the MADDPG agent
    agent = MADDPG(state_size, action_size, q_input_size, 1)

    # load network weights to MADDPG agent
    for i, ddpg_agent in enumerate(agent.maddpg_agent):
        ddpg_agent.actor_local.load_state_dict(torch.load('results/checkpoint_actor_agent%d.pth'%i))
        ddpg_agent.critic_local.load_state_dict(torch.load('results/checkpoint_critic_agent%d.pth'%i))

    # Play five episodes
    for i in range(5):                                         # play game for 5 episodes
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        while True:
            actions = agent.act(states, add_noise=False)       # get next action from the maddpg agent
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            states = env_info.vector_observations              # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            if np.any(dones):                                  # exit loop if episode finished
                break
        print('Total score this episode: {}'.format(np.max(scores)))

### Use the DDPG agent

The example code below shows how load and use the DDPG agent. The code is assumed to 
be run from the DDPG folder in this repo.

    from unityagents import UnityEnvironment
    import torch
    from agent import Agent
    import numpy as np

    ## Note: only run this cell if the environment is not loaded yet!

    # load the Tennis app
    env = UnityEnvironment(file_name="/data/Tennis_Linux_NoVis/Tennis")

    # get Tennis app info
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]          # reset the environment 
    states = env_info.vector_observations
    state_size = states.shape[1]                               # get state size
    action_size = brain.vector_action_space_size               # get action size
    q_input_size = int(2 * (state_size + action_size))         # get critic input size
    num_agents = len(env_info.agents)                          # get number of agents

    # instantiate the DDPG agent
    agent = Agent(state_size, action_size, random_seed=1)

    # load network weights to DDPG agent
    agent.actor_local.load_state_dict(torch.load('results/checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('results/checkpoint_critic.pth'))

    # Play 5 episodes
    n=5
    for i in range(n):                                         # play game for n episodes
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        while True:
                                                            # get next action from the ddpg agent
            actions = [agent.act(states[agent_i], add_noise=False) for agent_i in range(num_agents)]
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            states = env_info.vector_observations              # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            if np.any(dones):                                  # exit loop if episode finished
                break
        print('Total score this episode: {}'.format(np.max(scores)))
        

### Train the agent

It is also possible to train your own agent using either the MADDPG or the DDPG algorithm. 
For this purpose the notebooks `MADDPG/Tennis.ipynb` and `DDPG/Tenning_DDPG.ipynb` are provided.
Hyperparameters related to the neural networks and the (MA)DDPG algorithm
can be set in the file `hyperparameters.py` in the MADDPG and DDPG folders. 
Hyperparameters related to the training loop, such as the maximum number of steps 
in an episode or the maximum number of episodes to train, 
can be set in the notebooks itself. Network weights of the best performing agent are
saved in a folder which can be specified with the parameter `version` in the notebook.
Note that the agent continues training for another
few episodes after the acceptance criterium has been met, in order to get some feeling
of the stability of the training, and to try to increase the performance of the agent
beyond the acceptance criterium. 
