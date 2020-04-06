import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from RelationalModule import RelationalNetworks as rnet
from RelationalModule import ControlNetworks as cnet

class BoxWorldActor(nn.Module):
    """
    Use BoxWorldNet followed by a linear layer with log-softmax activation.
    """
    def __init__(self, action_space, **box_net_args):
        """
        Parameters
        ----------
        action_space: int
            Number of (discrete) possible actions to take
        **box_net_args: dict (optional)
            Dictionary of {'key':value} pairs valid for BoxWorldNet.
            Valid keys:
                in_channels: int (default 3)
                    Number of channels of the input image (e.g. 3 for RGB)
                n_kernels: int (default 24)
                    Number of features extracted for each pixel
                vocab_size: int (default 116)
                    Range of integer values of the raw pixels
                n_dim: int (default 3)
                    Embedding dimension for each pixel channel (1 channel for greyscale, 
                    3 for RGB)
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
        super(BoxWorldActor, self).__init__()
        self.boxnet = rnet.BoxWorldNet(**box_net_args)
        self.linear = nn.Linear(self.boxnet.n_features, action_space)
        
    def forward(self, state):
        out = self.boxnet(state)
        log_probs = F.log_softmax(self.linear(out), dim=1)
        return log_probs
        
class BoxWorldBasicCritic(nn.Module):
    """
    Use BoxWorldNet followed by a linear layer with a scalar output without
    activation function.
    """
    
    def __init__(self, **box_net_args):
        """
        Parameters
        ----------
        **box_net_args: dict (optional)
            Dictionary of {'key':value} pairs valid for BoxWorldNet
        """
        super(BoxWorldBasicCritic, self).__init__()
        self.boxnet = rnet.BoxWorldNet(**box_net_args)
        self.linear = nn.Linear(self.boxnet.n_features, 1)
    
    def forward(self, state):
        out = self.boxnet(state)
        V = self.linear(out)
        return V
    
class BoxWorldCritic(nn.Module):
    """
    Implements a generic critic for BoxWorld environment, 
    that can have 2 independent networks is twin=True. 
    """
    def __init__(self, twin=True, target=False, **box_net_args):
        """
        Parameters
        ----------
        twin: bool
            If True uses 2 critics
        target: bool
            If True, returns the minimum between the two critic's predictions
        **box_net_args: dict (optional)
            Dictionary of {'key':value} pairs valid for BoxWorldNet.
            Valid keys:
                in_channels: int (default 3)
                    Number of channels of the input image (e.g. 3 for RGB)
                n_kernels: int (default 24)
                    Number of features extracted for each pixel
                vocab_size: int (default 116)
                    Range of integer values of the raw pixels
                n_dim: int (default 3)
                    Embedding dimension for each pixel channel (1 channel for greyscale, 
                    3 for RGB)
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
        super(BoxWorldCritic, self).__init__()
        
        self.twin = twin
        self.target = target
        
        if twin:
            self.net1 = BoxWorldBasicCritic(**box_net_args)
            self.net2 = BoxWorldBasicCritic(**box_net_args)
        else:
            self.net = BoxWorldBasicCritic(**box_net_args)
        
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
        
        
### Control agent ###

class ControlActor(nn.Module):
    """
    Use ControlNet followed by a linear layer with log-softmax activation.
    """
    def __init__(self, action_space, **control_net_args):
        """
        Parameters
        ----------
        action_space: int
            Number of (discrete) possible actions to take
        **control_net_args: dict (optional)
            Dictionary of {'key':value} pairs valid for BoxWorldNet.
            Valid keys:
                in_channels: int (default 1)
                    Number of channels of the input image (e.g. 3 for RGB)
                n_kernels: int (default 24)
                    Number of features extracted for each pixel
                vocab_size: int (default 117)
                    Range of integer values of the raw pixels
                n_dim: int (default 3)
                    Embedding dimension for each pixel channel (1 channel for greyscale, 
                    3 for RGB)
                n_features: int (default 256)
                    Number of linearly projected features after positional encoding.
                    This is the number of features used during the PositionwiseFeedForward
                    (PFF) block
                hidden_dim: int (default 64)
                    Number of hidden units in PFF layers
                n_control_modules: int (default 4)
                    Number of PFF layers
                n_linears: int (default 4)
                    Number of feature-wise feed-forward layers
        """
        super(ControlActor, self).__init__()
        self.control_net = cnet.ControlNet(**control_net_args)
        self.linear = nn.Linear(self.control_net.n_features, action_space)
        
    def forward(self, state):
        out = self.control_net(state)
        log_probs = F.log_softmax(self.linear(out), dim=1)
        return log_probs
        
class ControlBasicCritic(nn.Module):
    """
    Use ControlNet followed by a linear layer with a scalar output without
    activation function.
    """
    
    def __init__(self, **control_net_args):
        """
        Parameters
        ----------
        **control_net_args: dict (optional)
            Dictionary of {'key':value} pairs valid for ControlNet
        """
        super(ControlBasicCritic, self).__init__()
        self.control_net = cnet.ControlNet(**control_net_args)
        self.linear = nn.Linear(self.control_net.n_features, 1)
    
    def forward(self, state):
        out = self.control_net(state)
        V = self.linear(out)
        return V
    
class ControlCritic(nn.Module):
    """
    Implements a generic critic for BoxWorld environment, 
    that can have 2 independent networks is twin=True. 
    """
    def __init__(self, twin=True, target=False, **control_net_args):
        """
        Parameters
        ----------
        twin: bool
            If True uses 2 critics
        target: bool
            If True, returns the minimum between the two critic's predictions
        **control_net_args: dict (optional)
            Dictionary of {'key':value} pairs valid for BoxWorldNet.
            Valid keys:
                in_channels: int (default 1)
                    Number of channels of the input image (e.g. 3 for RGB)
                n_kernels: int (default 24)
                    Number of features extracted for each pixel
                vocab_size: int (default 117)
                    Range of integer values of the raw pixels
                n_dim: int (default 3)
                    Embedding dimension for each pixel channel (1 channel for greyscale, 
                    3 for RGB)
                n_features: int (default 256)
                    Number of linearly projected features after positional encoding.
                    This is the number of features used during the PositionwiseFeedForward
                    (PFF) block
                hidden_dim: int (default 64)
                    Number of hidden units in PFF layers
                n_control_modules: int (default 4)
                    Number of PFF layers
                n_linears: int (default 4)
                    Number of feature-wise feed-forward layers
        """
        super(ControlCritic, self).__init__()
        
        self.twin = twin
        self.target = target
        
        if twin:
            self.net1 = ControlBasicCritic(**control_net_args)
            self.net2 = ControlBasicCritic(**control_net_args)
        else:
            self.net = ControlBasicCritic(**control_net_args)
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        
    
