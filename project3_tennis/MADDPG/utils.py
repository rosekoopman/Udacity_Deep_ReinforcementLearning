import os
import torch
import copy
import random
import numpy as np
import pandas as pd
from collections import namedtuple, deque
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """

        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    
def seeding(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def save_scores(all_scores, all_mean_scores, e_solved, version='.'):
    pd.DataFrame({'all_scores': all_scores,
                  'all_mean_scores': all_mean_scores,
                  'e_solved': e_solved}).to_csv(os.path.join(version,'scores.csv'))
    return


def plot_scores(all_scores, all_mean_scores, e_solved, version='.'):
    
    plt.plot(range(len(all_scores)), all_scores, label='score')
    plt.plot(range(len(all_mean_scores)), all_mean_scores, label='mean score')
    plt.axhline(0.5, linestyle='--', color='r', label='solved criterium')
    plt.arrow(e_solved, max(all_scores), 0, -(max(all_scores)-1), color='black', zorder=3, shape='full', width=30,
              head_width=150, head_length=0.2)
    plt.xlabel('Episode')
    plt.ylabel('(Mean) score')
    plt.title('Maximum Score = {0:.1f} \n Maximum Mean Score = {1:.3f} \n Solved at episode {2:d}'.format(max(all_scores),
                                                                                                          max(all_mean_scores),
                                                                                                          e_solved))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(version, 'scores.png'))
    plt.savefig(os.path.join(version, 'scores.pdf'))
    
    return