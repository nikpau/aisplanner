import torch
import torch.nn as nn
import torch.functional as F
from torch.distributions import Categorical

class Discriminator(nn.Module):
    def __init__(self, state_dim,n_actions):
        super().__init__()
        self.D = nn.Sequential(
            nn.Linear(state_dim + n_actions, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, state_action):
        return self.D(state_action)

class Learner(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def select_action(self, state: torch.Tensor):
        action_prob = self.actor(state)
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)
        return action, action_log_prob

    def evaluate(self, state: torch.Tensor, action):
        action_prob = self.actor(state)
        action_dist = Categorical(action_prob)
        action_log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        value = self.critic(state)
        return value, action_log_prob, entropy

    def forward(self):
        raise NotImplementedError