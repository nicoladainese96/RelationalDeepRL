import numpy as np
import copy
import torch 
import torch.nn as nn
import torch.nn.functional as F

debug = False

class ExtractEntities(nn.Module):
    """Parse raw RGB pixels into entieties (vectors of k_out dimensions)"""
    def __init__(self, k_out, k_in=1, vocab_size = 117, n_dim=3, kernel_size=2, stride=1, padding=0):
        super(ExtractEntities, self).__init__()
        assert k_out%2 == 0, "Please provide an even number of output kernels k_out"
        self.embed = nn.Embedding(vocab_size, n_dim)
        layers = []
        layers.append(nn.Conv2d(n_dim*k_in, k_out//2, kernel_size, stride, padding))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(k_out//2, k_out, kernel_size, stride, padding))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Accepts an input of shape (batch_size, k_in, linear_size, linear_size, )
        Returns a tensor of shape (batch_size, 2*k_out, linear_size, linear_size)
        """
        if debug:
            print("x.shape (before ExtractEntities): ", x.shape)
        if len(x.shape) <= 3:
            x = x.unsqueeze(0)
        x = self.embed(x)
        x = x.transpose(-1,-3)
        x = x.transpose(-1,-2).reshape(x.shape[0],-1,x.shape[-2],x.shape[-1])
        x = self.net(x)
        if debug:
            print("x.shape (ExtractEntities): ", x.shape)
        return x

class PositionalEncoding(nn.Module):
    """
    Adds two extra channels to the feature dimension, indicating the spatial 
    position (x and y) of each cell in the feature map using evenly spaced values
    between −1 and 1. Then projects the feature dimension to n_features through a 
    linear layer.
    """
    def __init__(self, n_kernels, n_features):
        super(PositionalEncoding, self).__init__()
        self.projection = nn.Linear(n_kernels + 2, n_features)

    def forward(self, x):
        """
        Accepts an input of shape (batch_size, linear_size, linear_size, n_kernels)
        Returns a tensor of shape (linear_size**2, batch_size, n_features)
        """
        x = self.add_encoding2D(x)
        if debug:
            print("x.shape (After encoding): ", x.shape)
        x = x.view(x.shape[0], x.shape[1],-1)
        if debug:
            print("x.shape (Before transposing and projection): ", x.shape)
        x = self.projection(x.transpose(2,1))
        x = x.transpose(1,0)
        
        if debug:
            print("x.shape (PositionalEncoding): ", x.shape)
        return x
    
    @staticmethod
    def add_encoding2D(x):
        x_ax = x.shape[-2]
        y_ax = x.shape[-1]
        
        x_lin = torch.linspace(-1,1,x_ax)
        xx = x_lin.repeat(x.shape[0],y_ax,1).view(-1, 1, y_ax, x_ax).transpose(3,2)
        
        y_lin = torch.linspace(-1,1,y_ax).view(-1,1)
        yy = y_lin.repeat(x.shape[0],1,x_ax).view(-1, 1, y_ax, x_ax).transpose(3,2)
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
        x = torch.cat((x,xx.to(device),yy.to(device)), axis=1)
        return x
    
class PositionwiseFeedForward(nn.Module):
    """
    Applies 2 linear layers with ReLU and dropout layers
    only after the first layer.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

def clones(module, N):
    "Produce N identical layers."
    return [copy.deepcopy(module) for _ in range(N)]

class ControlModule(nn.Module):
    """Implements an alternative to the relational module"""
    def __init__(self, n_kernels=24, n_features=256, hidden_dim=64, n_control_modules=4):
        """
        Parameters
        ----------
        n_kernels: int (default 24)
            Number of features extracted for each pixel
        n_features: int (default 256)
            Number of linearly projected features after positional encoding.
            This is the number of features used during the PositionwiseFeedForward
            (PFF) block
        hidden_dim: int (default 64)
            Number of hidden units in PFF layers
        n_control_modules: int (default 4)
            Number of PFF layers
        """
        super(ControlModule, self).__init__()
        
        enc_layer = PositionwiseFeedForward(n_features, hidden_dim)
        
        encoder_layers = clones(enc_layer, n_control_modules)
        
        self.net = nn.Sequential(
            PositionalEncoding(n_kernels, n_features),
            *encoder_layers)
        
        #if debug:
        #    print(self.net)
        
    def forward(self, x):
        """Expects an input of shape (batch_size, n_pixels, n_kernels)"""
        x = self.net(x)
        if debug:
            print("x.shape (RelationalModule): ", x.shape)
        return x

class FeaturewiseMaxPool(nn.Module):
    """Applies max pooling along a given axis of a tensor"""
    def __init__(self, pixel_axis):
        super(FeaturewiseMaxPool, self).__init__()
        self.max_along_axis = pixel_axis
        
    def forward(self, x):
        x, _ = torch.max(x, axis=self.max_along_axis)
        if debug:
            print("x.shape (FeaturewiseMaxPool): ", x.shape)
        return x
    
class ResidualLayer(nn.Module):
    """
    Implements residual layer. Use LayerNorm and ReLU activation before applying the layers.
    """
    def __init__(self, n_features, n_hidden):
        super(ResidualLayer, self).__init__()
        self.norm = nn.LayerNorm(n_features)
        self.w1 = nn.Linear(n_features, n_hidden)
        self.w2 = nn.Linear(n_hidden, n_features)

    def forward(self, x):
        out = F.relu(self.w1(self.norm(x)))
        out = F.relu(self.w2(out))
        return out + x
    
class ControlNet_v0(nn.Module):
    """
    Implements architecture for BoxWorld agent of the paper Relational Deep Reinforcement Learning.
    
    Architecture:
    - 2 Convolutional layers (2x2 kernel size, stride 1, first with 12 out channels, second with 24)
    - Positional Encoding layer (2 more channels encoding x,y pixels' positions) and then projecting the 26
      channels to 256
    - Control module
    - FeaturewiseMaxPool layer
    - Multi-layer Perceptron with some (defaul = 4) fully-connected layers
    
    """
    def __init__(self, in_channels=1, n_kernels=24, vocab_size=117, n_dim=3,
                 n_features=256, hidden_dim=64, n_control_modules=4, n_linears=4):
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
            This is the number of features used during the PositionwiseFeedForward
            (PFF) block
        hidden_dim: int (default 64)
            Number of hidden units in PFF layers
        n_control_modules: int (default 4)
            Number of PFF layers
        n_linears: int (default 4)
            Number of feature-wise feed-forward layers
        """
        super(ControlNet_v0, self).__init__()
        
        self.n_features = n_features
        
        MLP = clones( ResidualLayer(n_features,n_features), n_linears)
        
        self.net = nn.Sequential(
            ExtractEntities(n_kernels, in_channels, vocab_size, n_dim),
            ControlModule(n_kernels, n_features, hidden_dim, n_control_modules),
            FeaturewiseMaxPool(pixel_axis = 0),
            *MLP)
        
        if debug:
            print(self.net)
        
    def forward(self, x):
        x = self.net(x)
        if debug:
            print("x.shape (BoxWorldNet): ", x.shape)
        return x
           
        
class ControlNet(nn.Module):
    """
    Implements architecture for BoxWorld agent of the paper Relational Deep Reinforcement Learning.
    
    Architecture:
    - 2 Convolutional layers (2x2 kernel size, stride 1, first with 12 out channels, second with 24)
    - Positional Encoding layer (2 more channels encoding x,y pixels' positions) and then projecting the 26
      channels to 256
    - Control module
    - FeaturewiseMaxPool layer
    - Multi-layer Perceptron with some (defaul = 4) fully-connected layers
    
    """
    def __init__(self, vocab_size=117, n_dim=3, linear_size = 14, n_features=256):
        """
        Parameters
        ----------
        vocab_size: int (default 117)
            Range of integer values of the raw pixels
        n_dim: int (default 3)
            Embedding dimension for each pixel channel (1 channel for greyscale, 
            3 for RGB)
        n_features: int (default 256)
            Number of linearly projected features after positional encoding.
            This is the number of features used during the PositionwiseFeedForward
            (PFF) block
        n_linears: int (default 4)
            Number of feature-wise feed-forward layers
        """
        super(ControlNet, self).__init__()
        
        self.n_features = n_features

        self.embed = self.embed = nn.Embedding(vocab_size, n_dim)
        
        self.net = nn.Sequential( nn.Linear(n_dim*linear_size**2, n_features*n_dim),
                                  ResidualLayer(n_features*n_dim, n_features),
                                  nn.Linear(n_features*n_dim, n_features),
                                  ResidualLayer(n_features, n_features)
                                )
        
        if debug:
            print(self.net)
        
    def forward(self, x):
        if len(x.shape) <= 3:
            x = x.unsqueeze(0)
        x = self.embed(x)
        x = x.transpose(-1,-3)
        x = x.transpose(-1,-2).reshape(x.shape[0],-1)
        x = self.net(x)
        if debug:
            print("x.shape (BoxWorldNet): ", x.shape)
        return x       
        
        
        
        
        
        
