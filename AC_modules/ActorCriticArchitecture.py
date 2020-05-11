import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import itertools as it

debug = True

### Independent Actor Critic architectures ###

class Actor(nn.Module):
    def __init__(self, model, action_space, n_features, **HPs):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
                        model(n_features=n_features, **HPs),
                        nn.Linear(n_features, 256),
                        nn.ReLU(),
                        nn.Linear(256, action_space))
        
    def forward(self, state):
        log_probs = F.log_softmax(self.net(state), dim=1)
        return log_probs

class BaseCritic(nn.Module):
    def __init__(self, model, n_features, **HPs):
        super(BaseCritic, self).__init__()
        self.net = model(n_features=n_features, **HPs)
        self.linear =  nn.Sequential(
                        nn.Linear(n_features, 256),
                        nn.ReLU(),
                        nn.Linear(256, 1))
        
    def forward(self, state):
        V = self.linear(self.net(state))
        return V

class Critic(nn.Module):
    def __init__(self, model, n_features, twin=True, target=False, **HPs):
        super(Critic, self).__init__()
        
        self.twin = twin
        self.target = target
        
        if twin:
            self.net1 = BaseCritic(model, n_features, **HPs)
            self.net2 = BaseCritic(model, n_features, **HPs)
        else:
            self.net = BaseCritic(model, n_features, **HPs)
        
    def forward(self, state):
        if self.twin:
            v1 = self.net1(state)
            v2 = self.net2(state)
            if self.target:
                v = torch.min(v1, v2) 
            else:
                return v1, v2
        else:
            v = self.net(state)
            
        return v

### Shared ActorCritic architecture

class SharedActor(nn.Module):
    def __init__(self, action_space, n_features):
        super(SharedActor, self).__init__()
        self.linear = nn.Sequential(
                        nn.Linear(n_features, 256),
                        nn.ReLU(),
                        nn.Linear(256, action_space))

    def forward(self, shared_repr):
        log_probs = F.log_softmax(self.linear(shared_repr), dim=1)
        return log_probs
    
class SharedCritic(nn.Module):
    def __init__(self, n_features):
        super(SharedCritic, self).__init__()
        self.net = nn.Sequential(
                        nn.Linear(n_features, 256),
                        nn.ReLU(),
                        nn.Linear(256, 1))

    def forward(self, shared_repr):
        V = self.net(shared_repr)
        return V

class SharedActorCritic(nn.Module):
    def __init__(self, model,  action_space, n_features, tau, **HPs):
        super(SharedActorCritic, self).__init__()
        
        self.tau = tau

        self.shared_architecture = model(n_features=n_features, **HPs)
        self.actor = SharedActor(action_space, n_features)
        self.critic = SharedCritic(n_features)
        
        self.critic_target = BaseCritic(model, n_features, **HPs)

        
            
        #print("last check")
        
    def forward(self, state):
        shared_repr = self.shared_architecture(state)
        return shared_repr
    
    def pi(self, x, full_pass=True):
        if full_pass:
            shared_repr = self.forward(x)
        else:
            shared_repr = x
        log_probs = self.actor(shared_repr)
        return log_probs
    
    def V_critic(self, x, full_pass=True):
        if full_pass:
            shared_repr = self.forward(x)
        else:
            shared_repr = x
        V = self.critic(shared_repr)
        return V
    
    def V_target(self, x):
        V = self.critic_target(x)
        return V

    def init_target(self):
        #critic_param_gen = it.chain(self.shared_architecture.parameters(), self.critic.parameters())
        if debug: print("first check")
        for trg_params, params in zip(self.critic_target.net.parameters(), self.shared_architecture.parameters() ):
            if debug:
                print("trg_params ", trg_params.shape)
                print("params ", params.shape)
            trg_params.data.copy_(params.data)
            
        if debug: print("intermediate check")
        
        for trg_params, params in zip(self.critic_target.linear.parameters(), self.critic.parameters() ):
            if debug:
                print("trg_params ", trg_params.shape)
                print("params ", params.shape)
            trg_params.data.copy_(params.data)
        if debug: print("final check")
        
    def update_target(self):
        critic_param_gen = it.chain(self.shared_architecture.parameters(), self.critic.parameters())
        for trg_params, params in zip(self.critic_target.parameters(), critic_param_gen ):
            trg_params.data.copy_((1.-self.tau)*trg_params.data + self.tau*params.data)
            
class SharedActorCritic_no_trg(nn.Module):
    def __init__(self, model,  action_space, n_features, **HPs):
        super(SharedActorCritic_no_trg, self).__init__()

        self.shared_architecture = model(n_features=n_features, **HPs)
        self.actor = SharedActor(action_space, n_features)
        self.critic = SharedCritic(n_features)
        
    def forward(self, state):
        shared_repr = self.shared_architecture(state)
        return shared_repr
    
    def pi(self, x, full_pass=True):
        if full_pass:
            shared_repr = self.forward(x)
        else:
            shared_repr = x
        log_probs = self.actor(shared_repr)
        return log_probs
    
    def V_critic(self, x, full_pass=True):
        if full_pass:
            shared_repr = self.forward(x)
        else:
            shared_repr = x
        V = self.critic(shared_repr)
        return V
    