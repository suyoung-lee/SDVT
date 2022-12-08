"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import helpers as utl
import torch.optim as optim
from torch.distributions import Categorical

class PolicyResample(nn.Module):
    def __init__(self,
                 args,
                 state_dim,
                 latent_dim,
                 prob_dim,
                 ):

        super(PolicyResample, self).__init__()
        self.args = args
        self.affine1 = nn.Linear(state_dim + 2*latent_dim + prob_dim, 64)
        self.affine2 = nn.Linear(64, 2)
        self.log_probs = []
        self.return_list = []
        self.optimiser = optim.Adam([*self.parameters()], lr=0.001)

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def select_action(self, input):
        #input: state(40)+mu(5)+var(5)+prob(5) = 55
        probs = self.forward(input)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(torch.diagonal(m.log_prob(action)))
        return action

    def append_return_list(self, return_list):
        self.return_list.append(return_list)

    def update(self, ret_rms = None):
        policy_loss = []
        for log_prob, R in zip(self.log_probs, self.return_list):
            #select tasks that minimize return so it's not -log_prob * R
            if ret_rms is None:
                policy_loss.append(log_prob * (R))
            else:
                policy_loss.append(log_prob * (R - ret_rms.mean)/np.sqrt(ret_rms.var + 1e-8))

        self.optimiser.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimiser.step()
        self.log_probs = []
        self.return_list = []

