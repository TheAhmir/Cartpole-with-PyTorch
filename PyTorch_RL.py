import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
env = gym.make('CartPole-v1')
print("There are {} actions".format(env.action_space.n))

# Model
# can move either left or right
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fcl = nn.Linear(4,128) # 4 parameters as the observation space
        self.actor = nn.Linear(128, 2) # 2 actions
        self.critic = nn.Linear(128, 1)
        self.saved_actions = []
        self.rewards = []
    
    def forward(self,x):
        x = F.relu(self.fcl(x))
        action_prob = F.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)
        return action_prob, state_values

# decide whether to move left or right
def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m - Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()

# calculate losses and perform backprop
def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    returns = []
    
    for r in model.rewards[::-1]:
        R = r + 0.99 * R # gamma = 0.99
        returns.insert(0,R)
        
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    
    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_11_loss(value, torch.tensor([R])))
        
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    
    loss.backward()
    optimizer.step()
    
    del model.rewards[:]
    del model.saved_actions[:]
    
model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()

