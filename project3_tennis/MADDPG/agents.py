import numpy as np
import random
import os

from models import Actor, Critic
from utils import ReplayBuffer, OUNoise
from hyperparameters import *

import torch
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, q_input_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, fc1_units=ACTOR_FC1_UNITS, 
                                 fc2_units=ACTOR_FC2_UNITS).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, random_seed, fc1_units=ACTOR_FC1_UNITS, 
                                  fc2_units=ACTOR_FC2_UNITS).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=ACTOR_LR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(q_input_size, random_seed, fc1_units=CRITIC_FC1_UNITS, 
                                   fc2_units=CRITIC_FC2_UNITS).to(DEVICE)
        self.critic_target = Critic(q_input_size, random_seed, fc1_units=CRITIC_FC1_UNITS, 
                                    fc2_units=CRITIC_FC2_UNITS).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=CRITIC_LR, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def act(self, state, add_noise=True, noise_decay=1):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(dim=0).to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).squeeze().cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action = action + np.random.choice([-1, 1], action.shape) * noise_decay * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()


class MADDPG:
    def __init__(self, state_size, action_size, q_input_size, random_seed):
        super(MADDPG, self).__init__()

        self.maddpg_agent = [DDPGAgent(state_size, action_size, q_input_size, random_seed), 
                             DDPGAgent(state_size, action_size, q_input_size, random_seed)]
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = GAMMA
        self.tau = TAU
        self.batch_size = BATCH_SIZE
        
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        
    def reset(self):
        for ddpg_agent in self.maddpg_agent:
            ddpg_agent.reset()
    
    def act(self, states, add_noise=True, noise_decay=1):
        """ get actors from all agents in the MADDPG object in eval mode """

        return [ddpg_agent.act(state, add_noise, noise_decay) for ddpg_agent, state in zip(self.maddpg_agent, states)]
        
    def actor_local_all(self, states, agent_i):
        """get actions from all agents in the MADDPG object using the local networks"""
        
        # transform tensor of shape (batchsize, num_agents, statesize) to list where each element of the
        # list contains the states for one agent
        states = self._tensor_to_list(states)

        # get the actions based on the states. Detach the results for the agent(s) NOT under consideration
        # to save computation. Derivatives are not calculated in backprop.
        actions = [self.maddpg_agent[i].actor_local(state) if i == agent_i else
                   self.maddpg_agent[i].actor_local(state).detach()
                   for i, state in enumerate(states)]
        
        # transform back to tensor with shape (batchsize, num_agents, actionsize)
        actions = self._list_to_tensor(actions)
        
        return actions
    

    def actor_target_all(self, states):
        """get actions from all the agents in the MADDPG object using the target networks """
        
        # transform tensor of shape (batchsize, num_agents, statesize) to list where each element of the
        # list contains the states for one agent
        states = self._tensor_to_list(states)
        
        # get the actions based on the states
        actions = [agent.actor_target(state) for agent, state in zip(self.maddpg_agent, states)]
        
        # transform back to tensor with shape (batchsize, num_agents, actionsize)
        actions = self._list_to_tensor(actions)
        
        return actions
    
    def _tensor_to_list(self, input_tensor):
        """
        Transforms tensor of shape (batch_size, num_agents, datasize) to 
        list of size num_agents and elements of the type tensor with 
        size (batchsize, datasize)

        Tensor(batchsize, num_agents, datasize) --> [ Tensor(batchsize, datasize), Tensor(batchsize, datasize)]
        """
        return [t.squeeze() for t in torch.split(input_tensor.t(), 1, dim=0)]

    def _list_to_tensor(self, input_list):
        """
        Transforms a list of length num_agents and tensor elements with
        shape (batchsize, datasize) to a tensor with shape (batch_size, num_agents, datasize)

        [ Tensor(batchsize, datasize), Tensor(batchsize, datasize)] --> Tensor(batchsize, num_agents, datasize) 
        """
        return torch.stack(input_list, dim=1)

    def update(self, timestep):
        """update the critics and actors of all the agents """
        
        if len(self.memory.memory) > self.batch_size and timestep%LEARN_EVERY == 0:

            for _ in range(N_LEARN):

                # learn
                for agent_i in range(len(self.maddpg_agent)):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.gamma, agent_i)

                # update networks
                for ddpg_agent in self.maddpg_agent:
                    self.soft_update(ddpg_agent.actor_local, ddpg_agent.actor_target, self.tau)
                    self.soft_update(ddpg_agent.critic_local, ddpg_agent.critic_target, self.tau)
            
            
    def learn(self, experiences, gamma, agent_i):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, states_next, dones = experiences

        agent = self.maddpg_agent[agent_i]

        # ---------------------------- update critic ---------------------------- #
        # Need information from the entire environment, ie actions and states from all agents
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target_all(states_next)    # actor_target (all agents)
        q_input_target = torch.cat((states_next.view(self.batch_size, -1), actions_next.view(self.batch_size, -1)), dim=1)
        Q_targets_next = agent.critic_target(q_input_target) # critic_target (agent i)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards[:, agent_i].unsqueeze(dim=1) + (gamma * Q_targets_next * (1 - dones[:, agent_i].unsqueeze(dim=1)))
        # Compute critic loss
        q_input_expected = torch.cat((states.view(self.batch_size, -1), actions.view(self.batch_size, -1)), dim=1)
        Q_expected = agent.critic_local(q_input_expected)    # critic_local (agent_i) -- expected reward from Q function
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        if DO_GRADIENT_CLIP_CRITIC:
            torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local_all(states, agent_i)  # actor_local (all)
        q_input = torch.cat((states.view(self.batch_size, -1), actions_pred.view(self.batch_size, -1)), dim=1)
        actor_loss = -agent.critic_local(q_input).mean()      # critic_local (agent_i)
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def checkpoint(self, path='./'):
        for i, agent in enumerate(self.maddpg_agent):
            torch.save(agent.actor_local.state_dict(), os.path.join(path, 'checkpoint_actor_agent%d.pth'%i))
            torch.save(agent.critic_local.state_dict(), os.path.join(path, 'checkpoint_critic_agent%d.pth'%i))

