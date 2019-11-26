#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Categorical

class PPO(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(PPO, self).__init__()
        ###########################
        
        #For robot
        self.linear1 = nn.Linear(in_features=state_dim, out_features=64)
        self.relu1 = nn.ReLU(inplace=True)
        nn.init.orthogonal_(self.linear1.weight, np.sqrt(2))
        
        self.linear2 = nn.Linear(in_features = 64,out_features =64)
        self.relu2 = nn.ReLU(inplace=True)
        nn.init.orthogonal_(self.linear2.weight, np.sqrt(2))
        self.drop = nn.Dropout(p=0.2)
        
        self.pi_logits = nn.Linear(in_features=64,
                                   out_features=action_dim)
        nn.init.orthogonal_(self.pi_logits.weight, np.sqrt(0.01))
        
        self.value = nn.Linear(in_features=64,
                               out_features=1)
        
       
        
        '''self.conv1 = nn.Conv2d(in_channels=4,
                               out_channels=32,
                               kernel_size=8,
                               stride=4,
                               padding=0)
        nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
        
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=0)
        nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
        
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=0)
        nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))
        self.lin = nn.Linear(in_features=7 * 7 * 64,
                             out_features=512)
        nn.init.orthogonal_(self.lin.weight, np.sqrt(2))
        self.pi_logits = nn.Linear(in_features=512,
                                   out_features=4)
        nn.init.orthogonal_(self.pi_logits.weight, np.sqrt(0.01))

        self.value = nn.Linear(in_features=512,
                               out_features=1)'''
 
          
    
    def forward(self, x):
        #h: torch.Tensor

        '''h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.reshape((-1, 7 * 7 * 64))

        h = F.relu(self.lin(h))

        pi = Categorical(logits=self.pi_logits(h))
        value = self.value(h).reshape(-1)'''
        #For robot
        h=self.relu1(self.linear1(x))
        h=self.drop(self.relu2(self.linear2(h)))
        #change this in continuous case
        pi = Categorical(logits=self.pi_logits(h))
        value = self.value(h).reshape(-1)

        return pi, value
