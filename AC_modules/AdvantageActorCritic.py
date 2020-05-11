import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

from AC_modules.ActorCriticArchitecture import *

debug = False

class SharedAC(nn.Module):
    def __init__(self, model, action_space, n_features, gamma=0.99, 
                 tau = 1., H=1e-3, n_steps = 1, device='cpu',**HPs):
        """
        Parameters
        ----------
        model: PyTorch nn.Module class
            Network that outputs (batch_size, n_features)
        action_space: int
            Number of (discrete) possible actions to take
        n_features: int
            Number of features used for representing the state before the last layers
            (actor and critic linear layers)
        lr: float in [0,1]
            Learning rate
        gamma: float in [0,1] (default=0.99)
            Discount factor
        tau: float in [0,1] (default = 1.)
            Regulates how fast the critic_target gets updates, i.e. what percentage of the weights
            inherits from the critic. If tau=1., critic and critic_target are identical 
            at every step, if tau=0. critic_target is unchangable. 
            As a default this feature is disabled setting tau = 1, but if one wants to use it a good
            empirical value is 0.005.
        H: float (default 1e-2)
            Entropy multiplicative factor in actor's loss
        n_steps: int (default=1)
            Number of steps considered in TD update
        device: str in {'cpu','cuda'}
            Select if training agent with cpu or gpu. 
            FIXME: At the moment is gpu is present, it MUST use the gpu.
        """
        super(SharedAC, self).__init__()
        
        # Parameters
        self.gamma = gamma
        self.n_actions = action_space
        self.tau = tau
        self.H = H
        self.n_steps = n_steps
        
        # Buffers
        #self.optim_steps = 0 # counts the number of optimization steps
        #self.env_steps = 0 # counts the number of environment steps
        
        if debug: print("params and buffers check")
        self.AC = SharedActorCritic(model, action_space, n_features, tau, device=device, **HPs)
        if debug: print("architecture check")
        self.device = device 
        self.AC.to(self.device) 
        
        if debug:
            print("="*10 +" A2C HyperParameters "+"="*10)
            print("Discount factor: ", self.gamma)
            print("Action space: ", self.n_actions)
            print("Update critic target factor: ", self.tau)
            print("n_steps for TD: ", self.n_steps)
            print("Device used: ", self.device)
            print("\n\n"+"="*10 +" A2C Architecture "+"="*10)
            print("Actor Critic architecture: \n", self.AC)

    def get_action(self, state, return_log=False):
        state = torch.from_numpy(state).float().to(self.device)
        log_probs = self.AC.pi(state)
        probs = torch.exp(log_probs)
        action = Categorical(probs).sample().item()
        if return_log:
            return action, log_probs.view(-1)[action], probs
        else:
            return action
        
    def init_target(self):
        self.AC.init_target()
        
    def update_target(self):
        self.AC.update_target()
        
    def compute_ac_loss(self, rewards, log_probs, distributions, states, done, bootstrap=None): 
        ### Compute n-steps rewards, states, discount factors and done mask ###
        
        n_step_rewards = self.compute_n_step_rewards(rewards)
        if debug:
            print("n_step_rewards.shape: ", n_step_rewards.shape)
            print("rewards.shape: ", rewards.shape)
            print("n_step_rewards: ", n_step_rewards)
            print("rewards: ", rewards)
            print("bootstrap: ", bootstrap)
                
        if bootstrap is not None:
            done[bootstrap] = False 
        if debug:
            print("done.shape: (before n_steps)", done.shape)
            print("done: (before n_steps)", done)
        
        old_states = torch.tensor(states[:-1]).float().to(self.device)

        new_states, Gamma_V, done = self.compute_n_step_states(states, done)
        new_states = torch.tensor(new_states).float().to(self.device)

        if debug:
            print("done.shape: (after n_steps)", done.shape)
            print("Gamma_V.shape: ", Gamma_V.shape)
            print("done: (after n_steps)", done)
            print("Gamma_V: ", Gamma_V)
            print("old_states.shape: ", old_states.shape)
            print("new_states.shape: ", new_states.shape)
            
        ### Wrap variables into tensors ###
        
        done = torch.LongTensor(done.astype(int)).to(self.device)
        if debug: print("log_probs: ", log_probs)
            
        log_probs = torch.stack(log_probs).to(self.device)
        if debug: print("log_probs: ", log_probs)
            
        distributions = torch.stack(distributions, axis=0).to(self.device)
        mask = (distributions == 0).nonzero()
        distributions[mask[:,0], mask[:,1]] = 1e-5
        if debug: print("distributions: ", distributions)
            
        n_step_rewards = torch.tensor(n_step_rewards).float().to(self.device)
        Gamma_V = torch.tensor(Gamma_V).float().to(self.device)
        
        ### Update critic and then actor ###
        critic_loss = self.compute_critic_loss(n_step_rewards, new_states, old_states, done, Gamma_V)

        actor_loss, entropy = self.compute_actor_loss(n_step_rewards, log_probs, distributions, 
                                                       new_states, old_states, done, Gamma_V)

        return critic_loss, actor_loss, entropy
    
    def compute_critic_loss(self, n_step_rewards, new_states, old_states, done, Gamma_V):
        
        # Compute loss 
        if debug: print("Updating critic...")
        with torch.no_grad():
            V_trg = self.AC.V_target(new_states).squeeze()
            if debug:
                print("V_trg.shape (after critic): ", V_trg.shape)
            V_trg = (1-done)*Gamma_V*V_trg + n_step_rewards
            if debug:
                print("V_trg.shape (after sum): ", V_trg.shape)
            V_trg = V_trg.squeeze()
            if debug:
                print("V_trg.shape (after squeeze): ", V_trg.shape)
                print("V_trg.shape (after squeeze): ", V_trg)
            
        V = self.AC.V_critic(old_states).squeeze()
        if debug: 
            print("V.shape: ",  V.shape)
            print("V: ",  V)
        loss = F.mse_loss(V, V_trg)

        return loss
    
    def compute_actor_loss(self, n_step_rewards, log_probs, distributions, new_states, old_states, done, Gamma_V):
        
        # Compute gradient 
        if debug: print("Updating actor...")
        with torch.no_grad():
            V_pred = self.AC.V_critic(old_states).squeeze()
            V_trg = (1-done)*Gamma_V*self.AC.V_critic(new_states).squeeze()  + n_step_rewards
        
        A = V_trg - V_pred
        policy_gradient = - log_probs*A
        if debug:
            print("V_trg.shape: ",V_trg.shape)
            print("V_trg: ", V_trg)
            print("V_pred.shape: ",V_pred.shape)
            print("V_pred: ", V_pred)
            print("A.shape: ", A.shape)
            print("A: ", A)
            print("policy_gradient.shape: ", policy_gradient.shape)
            print("policy_gradient: ", policy_gradient)
        policy_grad = torch.mean(policy_gradient)
        if debug: print("policy_grad: ", policy_grad)
            
        # Compute negative entropy (no - in front)
        entropy = torch.mean(distributions*torch.log(distributions))
        if debug: print("Negative entropy: ", entropy)
        
        loss = policy_grad + self.H*entropy
        if debug: print("Actor loss: ", loss)
             
        return loss, entropy
                
    def compute_n_step_rewards(self, rewards):
        """
        Computes n-steps discounted reward padding with zeros the last elements of the trajectory.
        This means that the rewards considered are AT MOST n, but can be less for the last n-1 elements.
        """
        T = len(rewards)
        
        # concatenate n_steps zeros to the rewards -> they do not change the cumsum
        r = np.concatenate((rewards,[0 for _ in range(self.n_steps)])) 
        
        Gamma = np.array([self.gamma**i for i in range(r.shape[0])])
        
        # reverse everything to use cumsum in right order, then reverse again
        Gt = np.cumsum(r[::-1]*Gamma[::-1])[::-1]
        
        G_nstep = Gt[:T] - Gt[self.n_steps:] # compute n-steps discounted return
        
        Gamma = Gamma[:T]
        
        assert len(G_nstep) == T, "Something went wrong computing n-steps reward"
        
        n_steps_r = G_nstep / Gamma
        
        return n_steps_r
    
    def compute_n_step_states(self, states, done):
        """
        Computes n-steps target states (to be used by the critic as target values together with the
        n-steps discounted reward). For last n-1 elements the target state is the last one available.
        Adjusts also the `done` mask used for disabling the bootstrapping in the case of terminal states
        and returns Gamma_V, that are the discount factors for the target state-values, since they are 
        n-steps away (except for the last n-1 states, whose discount is adjusted accordingly).
        
        Return
        ------
        new_states, Gamma_V, done: arrays with first dimension = len(states)-1
        """
        
        # Compute indexes for (at most) n-step away states 
        
        n_step_idx = np.arange(len(states)-1) + self.n_steps
        diff = n_step_idx - len(states) + 1
        mask = (diff > 0)
        n_step_idx[mask] = len(states) - 1
        
        # Compute new states
        
        new_states = states[n_step_idx]
        
        # Compute discount factors
        
        pw = np.array([self.n_steps for _ in range(len(new_states))])
        pw[mask] = self.n_steps - diff[mask]
        Gamma_V = self.gamma**pw
        
        # Adjust done mask
        
        mask = (diff >= 0)
        done[mask] = done[-1]
        
        return new_states, Gamma_V, done
            
    
class IndependentAC(nn.Module):
    def __init__(self, model, action_space, n_features, gamma=0.99, tau = 1., 
                 twin=False, H=1e-3, n_steps = 1, device='cpu',**HPs):
        """
        Parameters
        ----------
        model: PyTorch nn.Module class
            Network that outputs (batch_size, n_features)
        action_space: int
            Number of (discrete) possible actions to take
        n_features: int
            Number of features used for representing the state before the last layers
            (actor and critic linear layers)
        lr: float in [0,1]
            Learning rate
        gamma: float in [0,1] (default=0.99)
            Discount factor
        twin: bool (default=False)
            Enables twin networks both for critic and critic_target
        tau: float in [0,1] (default = 1.)
            Regulates how fast the critic_target gets updates, i.e. what percentage of the weights
            inherits from the critic. If tau=1., critic and critic_target are identical 
            at every step, if tau=0. critic_target is unchangable. 
            As a default this feature is disabled setting tau = 1, but if one wants to use it a good
            empirical value is 0.005.
        H: float (default 1e-2)
            Entropy multiplicative factor in actor's loss
        n_steps: int (default=1)
            Number of steps considered in TD update
        device: str in {'cpu','cuda'}
            Select if training agent with cpu or gpu. 
            FIXME: At the moment is gpu is present, it MUST use the gpu.
        """
        super(IndependentAC, self).__init__()
        
        # Parameters
        self.gamma = gamma
        self.n_actions = action_space
        self.twin = twin 
        self.tau = tau
        self.H = H
        self.n_steps = n_steps
        
        # Buffers
        #self.optim_steps = 0 # counts the number of optimization steps
        #self.env_steps = 0 # counts the number of environment steps
        
        # ActorCritic architectures
        self.actor = Actor(model, action_space, n_features, device=device, **HPs)
        self.critic = Critic(model, n_features, twin=twin, device=device, **HPs)

        self.critic_trg = Critic(model, n_features, twin=twin, target=True, device=device, **HPs)

        # Init critic target identical to critic
        #for trg_params, params in zip(self.critic_trg.parameters(), self.critic.parameters()):
        #    trg_params.data.copy_(params.data)
        
        self.device = device 
        self.actor.to(self.device) 
        self.critic.to(self.device)
        #self.critic_trg.to(self.device)
        
        if debug:
            print("="*10 +" A2C HyperParameters "+"="*10)
            print("Discount factor: ", self.gamma)
            print("Action space: ", self.n_actions)
            print("Twin networks: ", self.twin)
            print("Update critic target factor: ", self.tau)
            print("n_steps for TD: ", self.n_steps)
            print("Device used: ", self.device)
            print("\n\n"+"="*10 +" A2C Architecture "+"="*10)
            print("Actor architecture: \n", self.actor)
            print("Critic architecture: \n",self.critic)
            #print("Critic target architecture: \n", self.critic_trg)
            
    def get_action(self, state, return_log=False):
        state = torch.from_numpy(state).float().to(self.device)
        log_probs = self.actor(state)
        probs = torch.exp(log_probs)
        action = Categorical(probs).sample().item()
        if return_log:
            return action, log_probs.view(-1)[action], probs
        else:
            return action
    
    def init_target(self):
        # Init critic target identical to critic
        for trg_params, params in zip(self.critic_trg.parameters(), self.critic.parameters()):
            trg_params.data.copy_(params.data)
            
    def update_target(self):
        for trg_params, params in zip(self.critic_trg.parameters(), self.critic.parameters()):
                trg_params.data.copy_((1.-self.tau)*trg_params.data + self.tau*params.data)
                
    def compute_ac_loss(self, rewards, log_probs, distributions, states, done, bootstrap=None): 
        ### Compute n-steps rewards, states, discount factors and done mask ###
        
        n_step_rewards = self.compute_n_step_rewards(rewards)
        if debug:
            print("n_step_rewards.shape: ", n_step_rewards.shape)
            print("rewards.shape: ", rewards.shape)
            print("n_step_rewards: ", n_step_rewards)
            print("rewards: ", rewards)
            print("bootstrap: ", bootstrap)
                
        if bootstrap is not None:
            done[bootstrap] = False 
        if debug:
            print("done.shape: (before n_steps)", done.shape)
            print("done: (before n_steps)", done)
        
        old_states = torch.tensor(states[:-1]).float().to(self.device)

        new_states, Gamma_V, done = self.compute_n_step_states(states, done)
        new_states = torch.tensor(new_states).float().to(self.device)

        if debug:
            print("done.shape: (after n_steps)", done.shape)
            print("Gamma_V.shape: ", Gamma_V.shape)
            print("done: (after n_steps)", done)
            print("Gamma_V: ", Gamma_V)
            print("old_states.shape: ", old_states.shape)
            print("new_states.shape: ", new_states.shape)
            
        ### Wrap variables into tensors ###
        
        done = torch.LongTensor(done.astype(int)).to(self.device)
        if debug: print("log_probs: ", log_probs)
            
        log_probs = torch.stack(log_probs).to(self.device)
        if debug: print("log_probs: ", log_probs)
            
        distributions = torch.stack(distributions, axis=0).to(self.device)
        mask = (distributions == 0).nonzero()
        distributions[mask[:,0], mask[:,1]] = 1e-5
        if debug: print("distributions: ", distributions)
            
        n_step_rewards = torch.tensor(n_step_rewards).float().to(self.device)
        Gamma_V = torch.tensor(Gamma_V).float().to(self.device)
        
        ### Update critic and then actor ###
        critic_loss = self.compute_critic_loss(n_step_rewards, new_states, old_states, done, Gamma_V)

        actor_loss, entropy = self.compute_actor_loss(n_step_rewards, log_probs, distributions, 
                                                       new_states, old_states, done, Gamma_V)

        return critic_loss, actor_loss, entropy
    
    def compute_critic_loss(self, n_step_rewards, new_states, old_states, done, Gamma_V):
        
        # Compute loss 
        if debug: print("Updating critic...")
        with torch.no_grad():
            V_trg = self.critic_trg(new_states).squeeze()
            if debug:
                print("V_trg.shape (after critic): ", V_trg.shape)
            V_trg = (1-done)*Gamma_V*V_trg + n_step_rewards
            if debug:
                print("V_trg.shape (after sum): ", V_trg.shape)
            V_trg = V_trg.squeeze()
            if debug:
                print("V_trg.shape (after squeeze): ", V_trg.shape)
                print("V_trg.shape (after squeeze): ", V_trg)
            
        if self.twin:
            V1, V2 = self.critic(old_states)
            if debug:
                print("V1.shape: ", V1.squeeze().shape)
                print("V1: ", V1)
            loss1 = 0.5*F.mse_loss(V1.squeeze(), V_trg)
            loss2 = 0.5*F.mse_loss(V2.squeeze(), V_trg)
            loss = loss1 + loss2
        else:
            V = self.critic(old_states).squeeze()
            if debug: 
                print("V.shape: ",  V.shape)
                print("V: ",  V)
            loss = F.mse_loss(V, V_trg)

        return loss
    
    def compute_actor_loss(self, n_step_rewards, log_probs, distributions, new_states, old_states, done, Gamma_V):
        
        # Compute gradient 
        if debug: print("Updating actor...")
        with torch.no_grad():
            if self.twin:
                V1, V2 = self.critic(old_states)
                V_pred = torch.min(V1.squeeze(), V2.squeeze())
                V1_new, V2_new = self.critic(new_states)
                V_new = torch.min(V1_new.squeeze(), V2_new.squeeze())
                V_trg = (1-done)*Gamma_V*V_new + n_step_rewards
            else:
                V_pred = self.critic(old_states).squeeze()
                V_trg = (1-done)*Gamma_V*self.critic(new_states).squeeze()  + n_step_rewards
        
        A = V_trg - V_pred
        policy_gradient = - log_probs*A
        if debug:
            print("V_trg.shape: ",V_trg.shape)
            print("V_trg: ", V_trg)
            print("V_pred.shape: ",V_pred.shape)
            print("V_pred: ", V_pred)
            print("A.shape: ", A.shape)
            print("A: ", A)
            print("policy_gradient.shape: ", policy_gradient.shape)
            print("policy_gradient: ", policy_gradient)
        policy_grad = torch.mean(policy_gradient)
        if debug: print("policy_grad: ", policy_grad)
            
        # Compute negative entropy (no - in front)
        entropy = torch.mean(distributions*torch.log(distributions))
        if debug: print("Negative entropy: ", entropy)
        
        loss = policy_grad + self.H*entropy
        if debug: print("Actor loss: ", loss)
             
        return loss, entropy
                
    def compute_n_step_rewards(self, rewards):
        """
        Computes n-steps discounted reward padding with zeros the last elements of the trajectory.
        This means that the rewards considered are AT MOST n, but can be less for the last n-1 elements.
        """
        T = len(rewards)
        
        # concatenate n_steps zeros to the rewards -> they do not change the cumsum
        r = np.concatenate((rewards,[0 for _ in range(self.n_steps)])) 
        
        Gamma = np.array([self.gamma**i for i in range(r.shape[0])])
        
        # reverse everything to use cumsum in right order, then reverse again
        Gt = np.cumsum(r[::-1]*Gamma[::-1])[::-1]
        
        G_nstep = Gt[:T] - Gt[self.n_steps:] # compute n-steps discounted return
        
        Gamma = Gamma[:T]
        
        assert len(G_nstep) == T, "Something went wrong computing n-steps reward"
        
        n_steps_r = G_nstep / Gamma
        
        return n_steps_r
    
    def compute_n_step_states(self, states, done):
        """
        Computes n-steps target states (to be used by the critic as target values together with the
        n-steps discounted reward). For last n-1 elements the target state is the last one available.
        Adjusts also the `done` mask used for disabling the bootstrapping in the case of terminal states
        and returns Gamma_V, that are the discount factors for the target state-values, since they are 
        n-steps away (except for the last n-1 states, whose discount is adjusted accordingly).
        
        Return
        ------
        new_states, Gamma_V, done: arrays with first dimension = len(states)-1
        """
        
        # Compute indexes for (at most) n-step away states 
        
        n_step_idx = np.arange(len(states)-1) + self.n_steps
        diff = n_step_idx - len(states) + 1
        mask = (diff > 0)
        n_step_idx[mask] = len(states) - 1
        
        # Compute new states
        
        new_states = states[n_step_idx]
        
        # Compute discount factors
        
        pw = np.array([self.n_steps for _ in range(len(new_states))])
        pw[mask] = self.n_steps - diff[mask]
        Gamma_V = self.gamma**pw
        
        # Adjust done mask
        
        mask = (diff >= 0)
        done[mask] = done[-1]
        
        return new_states, Gamma_V, done
           
    