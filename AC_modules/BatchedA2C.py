import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

from AC_modules.ActorCriticArchitecture import *

debug = False

class SharedA2C(nn.Module):
    def __init__(self, model, action_space, n_features, gamma=0.99, H=1e-3, n_steps = 1, device='cpu',**HPs):
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
        H: float (default 1e-2)
            Entropy multiplicative factor in actor's loss
        n_steps: int (default=1)
            Number of steps considered in TD update
        device: str in {'cpu','cuda'}
            Select if training agent with cpu or gpu. 
            FIXME: At the moment is gpu is present, it MUST use the gpu.
        """
        super(SharedA2C, self).__init__()
        
        # Parameters
        self.gamma = gamma
        self.n_actions = action_space
        self.H = H
        self.n_steps = n_steps
        
        if debug: print("params and buffers check")
        self.AC = SharedActorCritic_no_trg(model, action_space, n_features, device=device, **HPs)
        if debug: print("architecture check")
        self.device = device 
        self.AC.to(self.device) 
        
        if debug:
            print("="*10 +" A2C HyperParameters "+"="*10)
            print("Discount factor: ", self.gamma)
            print("Action space: ", self.n_actions)
            print("n_steps for TD: ", self.n_steps)
            print("Device used: ", self.device)
            print("\n\n"+"="*10 +" A2C Architecture "+"="*10)
            print("Actor Critic architecture: \n", self.AC)

    def get_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        #print("state.shape: ", state.shape)
        log_probs = self.AC.pi(state)
        #print("log_probs.shape: ", log_probs.shape)
        #print("log_probs: ", log_probs)
        probs = torch.exp(log_probs)
        action = Categorical(probs).sample()
        #print("action.shape: ", action.shape)
        action = action.detach().cpu().numpy()
        return action, log_probs[range(len(action)), action], probs

    def compute_ac_loss(self, rewards, log_probs, distributions, states, done, bootstrap, trg_states): 
        ### Compute n-steps rewards, states, discount factors and done mask ###
        
        n_step_rewards, episode_mask, n_steps_mask_b = self.compute_n_step_rewards(rewards, done)
        if debug:
            print("n_step_rewards.shape: ", n_step_rewards.shape)
            print("rewards.shape: ", rewards.shape)
            print("n_step_rewards: ", n_step_rewards)
            print("rewards: ", rewards)
            print("bootstrap: ", bootstrap)
        # seems to work
        done[bootstrap] = False 
        
        if debug:
            print("done.shape: (before n_steps)", done.shape)
            print("done: (before n_steps)", done)
        
        # merge batch and episode dimensions
        old_states = torch.tensor(states).float().to(self.device).reshape((-1,)+states.shape[2:])

        new_states, Gamma_V, done = self.compute_n_step_states(states, done, episode_mask, n_steps_mask_b)
        # merge batch and episode dimensions
        new_states = torch.tensor(new_states).float().to(self.device).reshape((-1,)+states.shape[2:])

        if debug:
            print("done.shape: (after n_steps)", done.shape)
            print("Gamma_V.shape: ", Gamma_V.shape)
            print("done: (after n_steps)", done)
            print("Gamma_V: ", Gamma_V)
            print("old_states.shape: ", old_states.shape)
            print("new_states.shape: ", new_states.shape)
            
        ### Wrap variables into tensors - merge batch and episode dimensions ###
        
        done = torch.LongTensor(done.astype(int)).to(self.device).reshape(-1)
            
        log_probs = torch.stack(log_probs).to(self.device).transpose(1,0).reshape(-1)
        if debug: print("log_probs.shape: ", log_probs.shape)
            
        distributions = torch.stack(distributions, axis=0).to(self.device).transpose(1,0).reshape(-1, self.n_actions)
        mask = (distributions == 0).nonzero()
        distributions[mask[:,0], mask[:,1]] = 1e-5
        if debug: print("distributions.shape: ", distributions.shape)
            
        n_step_rewards = torch.tensor(n_step_rewards).float().to(self.device).reshape(-1)
        Gamma_V = torch.tensor(Gamma_V).float().to(self.device).reshape(-1)
        
        ### Update critic and then actor ###
        critic_loss = self.compute_critic_loss(n_step_rewards, new_states, old_states, done, Gamma_V)

        actor_loss, entropy = self.compute_actor_loss(n_step_rewards, log_probs, distributions, 
                                                       new_states, old_states, done, Gamma_V)

        return critic_loss, actor_loss, entropy
    
    def compute_critic_loss(self, n_step_rewards, new_states, old_states, done, Gamma_V):
        
        # Compute loss 
        if debug: print("Updating critic...")
        with torch.no_grad():
            V_trg = self.AC.V_critic(new_states).squeeze()
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
                
    def compute_n_step_rewards(self, rewards, done):
        """
        Computes n-steps discounted reward padding with zeros the last elements of the trajectory.
        This means that the rewards considered are AT MOST n, but can be less for the last n-1 elements.
        """
        B = done.shape[0]
        T = done.shape[1]
        if debug:
            print("batch size: ", B)
            print("unroll len: ", T)
        
        
        # Compute episode mask (i-th row contains 1 if col j is in the same episode of col i, 0 otherwise)
        episode_mask = [[] for _ in range(B)]
        last = [-1 for _ in range(B)]
        xs, ys = np.nonzero(done)
        # Add done at the end of every batch to avoid exceptions -> not used in real target computations
        xs = np.concatenate([xs, np.arange(B)])
        ys = np.concatenate([ys, np.full(B, T-1)])
        for x, y in zip(xs, ys):
            m = [1 if (i > last[x] and i <= y) else 0 for i in range(T)]
            for _ in range(y-last[x]):
                episode_mask[x].append(m)
            last[x] = y
        episode_mask = np.array(episode_mask)
        
        # Compute n-steps mask and repeat it B times
        n_steps_mask = []
        for i in range(T):
            m = [1 if (j>=i and j<i+self.n_steps) else 0 for j in range(T)]
            n_steps_mask.append(m)
        n_steps_mask = np.array(n_steps_mask)
        n_steps_mask_b = np.repeat(n_steps_mask[np.newaxis,...] , B, axis=0)
        
        # Broadcast rewards to use multiplicative masks
        rewards_repeated = np.repeat(rewards[:,np.newaxis,:], T, axis=1)
        
        # Exponential discount factor
        Gamma = np.array([self.gamma**i for i in range(T)]).reshape(1,-1)
        if debug:
            print("Gamma.shape: ", Gamma.shape)
            print("rewards_repeated.shape: ", rewards_repeated.shape)
            print("episode_mask.shape: ", episode_mask.shape)
            print("n_steps_mask_b.shape: ", n_steps_mask_b.shape)
        n_steps_r = (Gamma*rewards_repeated*episode_mask*n_steps_mask_b).sum(axis=2)/Gamma
        return n_steps_r, episode_mask, n_steps_mask_b
    
    def compute_n_step_states(self, trg_states, done, episode_mask, n_steps_mask_b):
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
        
        B = done.shape[0]
        T = done.shape[1]
        V_mask = episode_mask*n_steps_mask_b
        b, x, y = np.nonzero(V_mask)
        V_trg_index = [[] for _ in range(B)]
        for b_i in range(B):
            valid_x = (b==b_i)
            for i in range(T):
                matching_x = (x==i)
                V_trg_index[b_i].append(y[valid_x*matching_x][-1])
        V_trg_index = np.array(V_trg_index)
        
        cols = np.array([], dtype=np.int)
        rows = np.array([], dtype=np.int)
        for i, v in enumerate(V_trg_index):
            cols = np.concatenate([cols, v], axis=0)
            row = np.full(V_trg_index.shape[1], i)
            rows = np.concatenate([rows, row], axis=0)
        new_states = trg_states[rows, cols].reshape(trg_states.shape)
        pw = V_trg_index - np.arange(V_trg_index.shape[1]) + 1
        Gamma_V = self.gamma**pw
        shifted_done = done[rows, cols].reshape(done.shape)
        return new_states, Gamma_V, shifted_done
            
    
class IndependentA2C(nn.Module):
    def __init__(self, model, action_space, n_features, gamma=0.99, twin=False, H=1e-3, n_steps = 1, device='cpu',**HPs):
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
        H: float (default 1e-2)
            Entropy multiplicative factor in actor's loss
        n_steps: int (default=1)
            Number of steps considered in TD update
        device: str in {'cpu','cuda'}
            Select if training agent with cpu or gpu. 
            FIXME: At the moment is gpu is present, it MUST use the gpu.
        """
        super(IndependentA2C, self).__init__()
        
        # Parameters
        self.gamma = gamma
        self.n_actions = action_space
        self.twin = twin 
        self.H = H
        self.n_steps = n_steps

        # ActorCritic architectures
        self.actor = Actor(model, action_space, n_features, device=device, **HPs)
        self.critic = Critic(model, n_features, twin=twin, device=device, **HPs)
        
        self.device = device 
        self.actor.to(self.device) 
        self.critic.to(self.device)
        
        if debug:
            print("="*10 +" A2C HyperParameters "+"="*10)
            print("Discount factor: ", self.gamma)
            print("Action space: ", self.n_actions)
            print("Twin networks: ", self.twin)
            print("n_steps for TD: ", self.n_steps)
            print("Device used: ", self.device)
            print("\n\n"+"="*10 +" A2C Architecture "+"="*10)
            print("Actor architecture: \n", self.actor)
            print("Critic architecture: \n",self.critic)
            
    def get_action(self, state, return_log=False):
        state = torch.from_numpy(state).float().to(self.device)
        log_probs = self.actor(state)
        probs = torch.exp(log_probs)
        action = Categorical(probs).sample().item()
        if return_log:
            return action, log_probs.view(-1)[action], probs
        else:
            return action
                
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
            if self.twin:
                V1_trg, V2_trg = self.critic(new_states)
                V_trg = torch.min(V1_trg, V2_trg).squeeze()
            else:
                V_trg = self.critic(new_states).squeeze()
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
           
    