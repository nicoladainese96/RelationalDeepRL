import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

from AC_networks import BoxWorldActor, BoxWorldCritic #custom module

class BoxWorldA2C():
    """
    Advantage Actor-Critic RL agent for BoxWorld environment described in the paper
    Relational Deep Reinforcement Learning.
    
    Notes
    -----
    * GPU implementation is still work in progress.
    * Always uses 2 separate networks for the critic,one that learns from new experience 
      (student/critic) and the other one (critic_target/teacher)that is more conservative 
      and whose weights are updated through an exponential moving average of the weights 
      of the critic, i.e.
          target.params = (1-tau)*target.params + tau* critic.params
    * In the case of Monte Carlo estimation the critic_target is never used
    * Possible to use twin networks for the critic and the critic target for improved 
      stability. Critic target is used for updates of both the actor and the critic and
      its output is the minimum between the predictions of its two internal networks.
      
    """ 
    
    def __init__(self, action_space, lr, gamma, TD=True, twin=False, tau = 1., device='cpu', debug=False, **box_net_args):
        """
        Parameters
        ----------
        action_space: int
            Number of (discrete) possible actions to take
        lr: float in [0,1]
            Learning rate
        gamma: float in [0,1]
            Discount factor
        TD: bool (default=True)
            If True, uses Temporal Difference for the critic's estimates
            Otherwise uses Monte Carlo estimation
        twin: bool (default=False)
            Enables twin networks both for critic and critic_target
        tau: float in [0,1] (default = 1.)
            Regulates how fast the critic_target gets updates, i.e. what percentage of the weights
            inherits from the critic. If tau=1., critic and critic_target are identical 
            at every step, if tau=0. critic_target is unchangable. 
            As a default this feature is disabled setting tau = 1, but if one wants to use it a good
            empirical value is 0.005.
        device: str in {'cpu','cuda'}
            Implemented, but GPU slower than CPU because it's difficult to optimize a RL agent without
            a replay buffer, that can be used only in off-policy algorithms.
        **box_net_args: dict (optional)
            Dictionary of {'key':value} pairs valid for BoxWorldNet.
            Valid keys:
                in_channels: int (default 3)
                    Number of channels of the input image (e.g. 3 for RGB)
                n_kernels: int (default 24)
                    Number of features extracted for each pixel
                n_features: int (default 256)
                    Number of linearly projected features after positional encoding.
                    This is the number of features used during the Multi-Headed Attention
                    (MHA) blocks
                n_heads: int (default 4)
                    Number of heades in each MHA block
                n_attn_modules: int (default 2)
                    Number of MHA blocks
                n_linears: int (default 4)
                    Number of fully-connected layers after the FeaturewiseMaxPool layer
        """
        
        self.gamma = gamma
        self.lr = lr
        
        self.n_actions = action_space
        self.TD = TD
        self.twin = twin 
        self.tau = tau
        
        self.actor = BoxWorldActor(action_space, **box_net_args)
        self.critic = BoxWorldCritic(twin, **box_net_args)
        
        if self.TD:
            self.critic_trg = BoxWorldCritic(twin, target=True, **box_net_args)

            # Init critic target identical to critic
            for trg_params, params in zip(self.critic_trg.parameters(), self.critic.parameters()):
                trg_params.data.copy_(params.data)
            
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        self.device = device 
        self.actor.to(self.device) 
        self.critic.to(self.device)
        if self.TD:
            self.critic_trg.to(self.device)
        
        if debug:
            print("="*10 +" A2C HyperParameters "+"="*10)
            print("Discount factor: ", self.gamma)
            print("Learning rate: ", self.lr)
            print("Action space: ", self.n_actions)
            print("Temporal Difference learning: ", self.TD)
            print("Twin networks: ", self.twin)
            print("Update critic target factor: ", self.tau)
            print("Device used: ", self.device)
            print("\n\n"+"="*10 +" A2C Architecture "+"="*10)
            print("Actor architecture: \n", self.actor)
            print("Critic architecture: \n",self.critic)
            print("Critic target architecture: ")
            if self.TD:
                print(self.critic_trg)
            else:
                print("Not used")
        
    def get_action(self, state, return_log=False):
        log_probs = self.forward(state)
        dist = torch.exp(log_probs)
        probs = Categorical(dist)
        action =  probs.sample().item()
        if return_log:
            return action, log_probs.view(-1)[action]
        else:
            return action
    
    def forward(self, state):
        """
        Makes a tensor out of a numpy array state and then forward
        it with the actor network.
        
        Parameters
        ----------
        state:
            If self.discrete is True state.shape = (episode_len,)
            Otherwise state.shape = (episode_len, observation_space)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        log_probs = self.actor(state)
        return log_probs
    
    def update(self, *args):
        if self.TD:
            critic_loss, actor_loss = self.update_TD(*args)
        else:
            critic_loss, actor_loss = self.update_MC(*args)
        
        return critic_loss, actor_loss
    
    def update_TD(self, rewards, log_probs, states, done, bootstrap=None):   
        
        ### Wrap variables into tensors ###
        
        old_states = torch.tensor(states[:,:-1]).float().to(self.device)
        new_states = torch.tensor(states[:,1:]).float().to(self.device)
            
        if bootstrap is not None:
            done[bootstrap] = False 
        done = torch.LongTensor(done.astype(int)).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        
        ### Update critic and then actor ###
        critic_loss = self.update_critic_TD(rewards, new_states, old_states, done)
        actor_loss = self.update_actor_TD(rewards, log_probs, new_states, old_states, done)
        
        return critic_loss, actor_loss
    
    def update_critic_TD(self, rewards, new_states, old_states, done):
        
        # Compute loss 
        
        with torch.no_grad():
            V_trg = self.critic_trg(new_states).squeeze()
            V_trg = (1-done)*self.gamma*V_trg + rewards
            V_trg = V_trg.squeeze()
            
        if self.twin:
            #print(self.critic(old_states))
            V1, V2 = self.critic(old_states)
            loss1 = 0.5*F.mse_loss(V1.squeeze(), V_trg)
            loss2 = 0.5*F.mse_loss(V2.squeeze(), V_trg)
            loss = loss1 + loss2
        else:
            V = self.critic(old_states).squeeze()
            loss = F.mse_loss(V, V_trg)
        
        # Backpropagate and update
        
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        
        # Update critic_target: (1-tau)*old + tau*new
        
        for trg_params, params in zip(self.critic_trg.parameters(), self.critic.parameters()):
                trg_params.data.copy_((1.-self.tau)*trg_params.data + self.tau*params.data)
        
        return loss.item()
    
    def update_actor_TD(self, rewards, log_probs, new_states, old_states, done):
        
        # Compute gradient 
        
        if self.twin:
            V1, V2 = self.critic(old_states)
            V_pred = torch.min(V1.squeeze(), V2.squeeze())
            V1_new, V2_new = self.critic(new_states)
            V_new = torch.min(V1_new.squeeze(), V2_new.squeeze())
            V_trg = (1-done)*self.gamma*V_new + rewards
        else:
            V_pred = self.critic(old_states).squeeze()
            V_trg = (1-done)*self.gamma*self.critic(new_states).squeeze()  + rewards
            
        A = V_trg - V_pred
        policy_gradient = - log_probs*A
        policy_grad = torch.sum(policy_gradient)
 
        # Backpropagate and update
    
        self.actor_optim.zero_grad()
        policy_grad.backward()
        self.actor_optim.step()
        
        return policy_grad.item()
    
    def update_MC(self, rewards, log_probs, states, done, bootstrap=None):   
        
        ### Compute MC discounted returns ###
        
        if bootstrap is not None:
            
            if bootstrap[-1] == True:
            
                last_state = torch.tensor(states[0,-1,:]).float().to(self.device).view(1,-1)
                
                if self.twin:
                    V1, V2 = self.critic(last_state)
                    V_bootstrap = torch.min(V1, V2).cpu().detach().numpy().reshape(1,)
                else:
                    V_bootstrap = self.critic(last_state).cpu().detach().numpy().reshape(1,)
 
                rewards = np.concatenate((rewards, V_bootstrap))
                
        Gamma = np.array([self.gamma**i for i in range(rewards.shape[0])])
        # reverse everything to use cumsum in right order, then reverse again
        Gt = np.cumsum(rewards[::-1]*Gamma[::-1])[::-1]
        # Rescale so that present reward is never discounted
        discounted_rewards =  Gt/Gamma
        
        if bootstrap is not None:
            if bootstrap[-1] == True:
                discounted_rewards = discounted_rewards[:-1] # drop last

        ### Wrap variables into tensors ###
        
        dr = torch.tensor(discounted_rewards).float().to(self.device) 
        
        old_states = torch.tensor(states[:,:-1]).float().to(self.device)
        new_states = torch.tensor(states[:,1:]).float().to(self.device)
            
        done = torch.LongTensor(done.astype(int)).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        
        ### Update critic and then actor ###
        
        critic_loss = self.update_critic_MC(dr, old_states)
        actor_loss = self.update_actor_MC(dr, log_probs, old_states)
        
        return critic_loss, actor_loss
    
    def update_critic_MC(self, dr, old_states):

        # Compute loss
        
        if self.twin:
            V1, V2 = self.critic(old_states)
            V_pred = torch.min(V1.squeeze(), V2.squeeze())
        else:
            V_pred = self.critic(old_states).squeeze()
            
        loss = F.mse_loss(V_pred, dr)
        
        # Backpropagate and update
        
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        
        return loss.item()
    
    def update_actor_MC(self, dr, log_probs, old_states):
        
        # Compute gradient 
        
        if self.twin:
            V1, V2 = self.critic(old_states)
            V_pred = torch.min(V1.squeeze(), V2.squeeze())
        else:
            V_pred = self.critic(old_states).squeeze()
            
        A = dr - V_pred
        policy_gradient = - log_probs*A
        policy_grad = torch.sum(policy_gradient)
 
        # Backpropagate and update
    
        self.actor_optim.zero_grad()
        policy_grad.backward()
        self.actor_optim.step()
        
        return policy_grad.item()

    