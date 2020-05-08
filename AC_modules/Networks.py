import numpy as np
import copy
import torch 
import torch.nn as nn
import torch.nn.functional as F

from AC_modules.Layers import *

debug = False

class CoordinateNet(nn.Module):
    def __init__(self, observation_space, n_features, hiddens=[64,32], device=None):
        super(CoordinateNet, self).__init__()
        
        layers = []
        layers.append(nn.Linear(observation_space, hiddens[0]))
        layers.append(nn.ReLU())
        for i in range(0,len(hiddens)-1):
            layers.append(nn.Linear(hiddens[i], hiddens[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hiddens[-1], n_features))
        layers.append(nn.ReLU())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, state):
        if debug: 
            print("state.shape: ", state.shape)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        return self.net(state)
    
class MultiplicativeNet(nn.Module):
    def __init__(self, in_channels=3, info_channels=2, mask_channels=2, device=None):
        super(MultiplicativeNet, self).__init__()
        
        out_channels = info_channels*mask_channels
        self.out_channels = out_channels
        
        self.pos_enc = PosEncoding(device)
        self.multi_layer = MultiplicativeLayer(in_channels+2, info_channels, mask_channels)
        self.MLP = nn.Sequential(
                                nn.Linear(out_channels, out_channels),
                                nn.ReLU(),
                                nn.Linear(out_channels, out_channels),
                                nn.ReLU(),
                                nn.Linear(out_channels, out_channels),
                                nn.Sigmoid()
                                )
        
    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        x = self.pos_enc(x)
        if debug: print("x.shape (pos enc): ", x.shape)
            
        x = self.multi_layer(x)
        if debug: print("x.shape (multi layer): ", x.shape)
        
        x = x.reshape((x.shape[0], x.shape[1], -1))
        if debug: print("x.shape (after rehsape): ", x.shape)
        
        x, _ = torch.max(x, axis=-1)
        if debug: print("x.shape (after max): ", x.shape)
        
        x = self.MLP(x)
        if debug: print("x.shape (after MLP): ", x.shape)
            
        return x
    
class MultiplicativeConvNet(nn.Module):
    
    def __init__(self, linear_size, in_channels=3, info_channels=6, mask_channels=4, hidden_channels=12, out_channels=[12,24], 
                 padding=1, max_pool_size=2, n_features=64, residual_hidden_dim=64, n_residual_layers=2, 
                 version='v1', plug_off=False, device=None):
        
        super(MultiplicativeConvNet, self).__init__()

        n_multi_blocks = len(out_channels)
        self.out_channels = out_channels[-1]
        self.out_size = self.compute_out_size(linear_size, n_multi_blocks, max_pool_size, padding)
        self.n_features = n_features
        if debug:
            print("Out channels after forward1: ", self.out_channels)
            print("Linear size after forward1: ", self.out_size)
            
        if plug_off:
            self.forward1 = nn.Sequential(PosEncoding(device), 
                                          nn.Conv2d(in_channels+2, out_channels[-1], 3, 1, 1),
                                          nn.ReLU(),
                                          ResidualConvolutional(linear_size, out_channels[-1], hidden_channels),
                                          nn.ReLU(),
                                          ResidualConvolutional(linear_size, out_channels[-1], hidden_channels),
                                          nn.ReLU()
                                          )
            
        else:
            multi_blocks = nn.ModuleList(
                [MultiplicativeBlock(in_channels+2, out_channels[0], info_channels, 
                                     mask_channels, hidden_channels, padding=padding, version=version)]+
                [MultiplicativeBlock(out_channels[i], out_channels[i+1], info_channels, 
                                     mask_channels, hidden_channels, padding=padding, version=version) 
                 for i in range(n_multi_blocks-1)])
            
            self.forward1 = nn.Sequential( PosEncoding(device), *multi_blocks)
            
        self.maxpool = nn.MaxPool2d(max_pool_size)

        residual_MLP = nn.ModuleList([ResidualLayer(n_features, residual_hidden_dim)
                                      for _ in range(n_residual_layers)])
        self.forward2 = nn.Sequential(nn.Linear(self.out_channels*self.out_size**2, n_features), *residual_MLP)
    
    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        x = self.forward1(x)
        if debug: print("x.shape (after forward1): ", x.shape)
            
        x = self.maxpool(x)
        if debug: print("x.shape (after maxpool): ", x.shape)
            
        x = x.reshape((x.shape[0],-1))
        if debug: print("x.shape (after rehsape): ", x.shape)
        
        x = self.forward2(x)
        if debug: print("x.shape (after residual MLP): ", x.shape)
            
        return x
    
    @staticmethod
    def compute_out_size(linear_size, n_multi_blocks, max_pool_size, padding):
        size = (linear_size - (2-2*padding)*n_multi_blocks) // max_pool_size
        return size
    
class BoxWorldNet_v0(nn.Module):
    """
    Implements architecture for BoxWorld agent of the paper Relational Deep Reinforcement Learning.
    
    Architecture:
    - 2 Convolutional layers (2x2 kernel size, stride 1, first with 12 out channels, second with 24)
    - Positional Encoding layer (2 more channels encoding x,y pixels' positions) and then projecting the 26
      channels to 256
    - Relational module, with one or more attention blocks (MultiheadedAttention + PositionwiseFeedForward)
    - FeaturewiseMaxPool layer
    - Multi-layer Perceptron with some (defaul = 4) fully-connected layers
    
    """
    def __init__(self, in_channels=1, n_kernels=24, vocab_size = 117, n_dim=3,
                 n_features=256, n_heads=4, n_attn_modules=2, n_linears=4, max_pool=True, linear_size=14, device=None):
        """
        Parameters
        ----------
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
            This is the number of features used during the Multi-Headed Attention
            (MHA) blocks
        n_heads: int (default 4)
            Number of heades in each MHA block
        n_attn_modules: int (default 2)
            Number of MHA blocks
        n_linears: int (default 4)
            Number of fully-connected layers after the FeaturewiseMaxPool layer
        """
        super(BoxWorldNet_v0, self).__init__()
        
        self.n_features = n_features

        MLP = nn.ModuleList([ResidualLayer(n_features, n_features) for _ in range(n_linears)])
        self.process_input = ExtractEntities(n_kernels, in_channels, vocab_size, n_dim)
        
        if max_pool:
            self.net = nn.Sequential(
                RelationalModule(n_kernels, n_features, n_heads, n_attn_modules, device=device),
                FeaturewiseMaxPool(pixel_axis = 0),
                *MLP)
        else:
            self.net = nn.Sequential(
                RelationalModule(n_kernels, n_features, n_heads, n_attn_modules, device=device),
                FeaturewiseProjection(int((linear_size-2)**2)),
                *MLP)
        
        if debug:
            print(self.net)
        
    def forward(self, state):
        x = self.process_input(state)
        x = self.net(x)
        if debug:
            print("x.shape (BoxWorldNet): ", x.shape)
        return x
    
    
class BoxWorldNet(nn.Module):
    """
    Implements architecture for BoxWorld agent of the paper Relational Deep Reinforcement Learning.
    
    Architecture:
    - 2 Convolutional layers (2x2 kernel size, stride 1, first with 12 out channels, second with 24)
    - Positional Encoding layer (2 more channels encoding x,y pixels' positions) and then projecting the 26
      channels to 256
    - Relational module, with one or more attention blocks (MultiheadedAttention + PositionwiseFeedForward)
    - FeaturewiseMaxPool layer
    - Multi-layer Perceptron with some (defaul = 4) fully-connected layers
    
    """
    def __init__(self, in_channels=3, n_kernels=24, n_features=32, n_heads=2, 
                 n_attn_modules=4, feature_hidden_dim=64, feature_n_residuals=4, device=None):
        """
        Parameters
        ----------
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
            This is the number of features used during the Multi-Headed Attention
            (MHA) blocks
        n_heads: int (default 4)
            Number of heades in each MHA block
        n_attn_modules: int (default 2)
            Number of MHA blocks
        n_linears: int (default 4)
            Number of fully-connected layers after the FeaturewiseMaxPool layer
        """
        super(BoxWorldNet, self).__init__()
        
        self.n_features = n_features

        MLP = nn.ModuleList([ResidualLayer(n_features, feature_hidden_dim) for _ in range(feature_n_residuals)])
        
        self.net = nn.Sequential(
            Convolution(k_in=in_channels, k_out=n_kernels),
            RelationalModule(n_kernels, n_features, n_heads, n_attn_modules, device=device),
            FeaturewiseMaxPool(pixel_axis = 0),
            *MLP)

        if debug:
            print(self.net)
        
    def forward(self, x):
        x = self.net(x)
        if debug:
            print("x.shape (BoxWorldNet): ", x.shape)
        return x
        
        
        
class OheNet(nn.Module):
    def __init__(self, map_size, k_in=3, k_out=24, n_features=32, pixel_hidden_dim=128, 
                 pixel_n_residuals=4, feature_hidden_dim=64, feature_n_residuals=4, device=None):
        
        super(OheNet, self).__init__()
        
        self.n_features = n_features
        
        self.OHE_conv = Convolution(k_in=k_in, k_out=k_out)
        self.pos_enc = PositionalEncoding(n_kernels=k_out, n_features=n_features, device=device)

        pixel_res_layers = nn.ModuleList([ResidualLayer(map_size**2, pixel_hidden_dim) for _ in range(pixel_n_residuals)])
        self.pixel_res_block = nn.Sequential(*pixel_res_layers)

        self.maxpool = FeaturewiseMaxPool(pixel_axis=2)

        feature_res_layers = nn.ModuleList([ResidualLayer(n_features, feature_hidden_dim) for _ in range(feature_n_residuals)])
        self.feature_res_block = nn.Sequential(*feature_res_layers)
        
    def forward(self, x):
        """ Input shape (batch_dim, k_in, map_size+2, map_size+2) """
        
        x = self.OHE_conv(x)
        if debug: print("conv_state.shape: ", x.shape)
            
        x = self.pos_enc(x)
        if debug: print("After positional enc + projection: ", x.shape)
            
        x = x.permute(1,2,0)
        if debug: print("x.shape: ", x.shape)
            
        x = self.pixel_res_block(x) # Interaction between pixels feature-wise
        if debug: print("x.shape: ", x.shape)
            
        x = self.maxpool(x) # Feature-wise maxpooling
        if debug: print("x.shape: ", x.shape)
            
        x = self.feature_res_block(x) # Interaction between features -> final representation
        if debug: print("x.shape: ", x.shape)
        
        return x     
    
class GatedBoxWorldNet(nn.Module):
    """
    Implements architecture for BoxWorld agent of the paper Relational Deep Reinforcement Learning.
    
    Architecture:
    - 2 Convolutional layers (2x2 kernel size, stride 1, first with 12 out channels, second with 24)
    - Positional Encoding layer (2 more channels encoding x,y pixels' positions) and then projecting the 26
      channels to 256
    - Relational module, with one or more attention blocks (MultiheadedAttention + PositionwiseFeedForward)
    - FeaturewiseMaxPool layer
    - Multi-layer Perceptron with some (defaul = 4) fully-connected layers
    
    """
    def __init__(self, in_channels=3, n_kernels=24, n_features=32, n_heads=2, 
                 n_attn_modules=4, feature_hidden_dim=64, feature_n_residuals=4, device=None):
        """
        Parameters
        ----------
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
            This is the number of features used during the Multi-Headed Attention
            (MHA) blocks
        n_heads: int (default 4)
            Number of heades in each MHA block
        n_attn_modules: int (default 2)
            Number of MHA blocks
        n_linears: int (default 4)
            Number of fully-connected layers after the FeaturewiseMaxPool layer
        """
        super(GatedBoxWorldNet, self).__init__()
        
        self.n_features = n_features

        MLP = nn.ModuleList([ResidualLayer(n_features, feature_hidden_dim) for _ in range(feature_n_residuals)])
        
        self.net = nn.Sequential(
            Convolution(k_in=in_channels, k_out=n_kernels),
            GatedRelationalModule(n_kernels, n_features, n_heads, n_attn_modules, device=device),
            FeaturewiseMaxPool(pixel_axis = 0),
            *MLP)
   
        if debug:
            print(self.net)
        
    def forward(self, x):
        x = self.net(x)
        if debug:
            print("x.shape (BoxWorldNet): ", x.shape)
        return x