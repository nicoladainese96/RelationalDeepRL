import numpy as np
import copy
import torch 
import torch.nn as nn
import torch.nn.functional as F

debug = False

class ExtractEntities(nn.Module):
    """Parse raw RGB pixels into entieties (vectors of k_out dimensions)"""
    def __init__(self, k_out, k_in=1, kernel_size=2, stride=1, padding=0):
        super(ExtractEntities, self).__init__()
        assert k_out%2 == 0, "Please provide an even number of output kernels k_out"
        layers = []
        layers.append(nn.Conv2d(k_in, k_out//2, kernel_size, stride, padding))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(k_out//2, k_out, kernel_size, stride, padding))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Accepts an input of shape (batch_size, linear_size, linear_size, k_in)
        Returns a tensor of shape (batch_size, linear_size, linear_size, 2*k_out)
        """
        if debug:
            print("x.shape (before ExtractEntities): ", x.shape)
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
        
class AttentionBlock(nn.Module):
    def __init__(self, n_features, n_heads, n_hidden=64, dropout=0.1):
        """
        Args:
          n_features: Number of input and output features. (d_model)
          n_heads: Number of attention heads in the Multi-Head Attention.
          n_hidden: Number of hidden units in the Feedforward (MLP) block. (d_k)
          dropout: Dropout rate after the first layer of the MLP and the two skip connections.
        """
        super(AttentionBlock, self).__init__()
        self.norm = nn.LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(n_features, n_heads, dropout)
        self.ff = PositionwiseFeedForward(n_features, n_hidden, dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
          x of shape (n_pixels**2, batch_size, n_features): Input sequences.
          mask of shape (batch_size, max_seq_length): Boolean tensor indicating which elements of the input
              sequences should be ignored.
        
        Returns:
          z of shape (max_seq_length, batch_size, n_features): Encoded input sequence.

        Note: All intermediate signals should be of shape (n_pixels**2, batch_size, n_features).
        """

        attn_output, attn_output_weights = self.attn(x,x,x, key_padding_mask=mask) # MHA step
        x_norm = self.dropout(self.norm(attn_output + x)) # add and norm
        z = self.ff(x_norm) # FF step
        return self.dropout(self.norm(z)) # add and norm

def clones(module, N):
    "Produce N identical layers."
    return [copy.deepcopy(module) for _ in range(N)]

class RelationalModule(nn.Module):
    """Implements the relational module from paper Relational Deep Reinforcement Learning"""
    def __init__(self, n_kernels=24, n_features=256, n_heads=4, n_attn_modules=2):
        """
        Parameters
        ----------
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
        """
        super(RelationalModule, self).__init__()
        
        enc_layer = AttentionBlock(n_features, n_heads, n_hidden=64, dropout=0.1)
        
        encoder_layers = clones(enc_layer, n_attn_modules)
        
        self.net = nn.Sequential(
            PositionalEncoding(n_kernels, n_features),
            *encoder_layers)
        
        if debug:
            print(self.net)
        
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
    def __init__(self, in_channels=1, n_kernels=24, n_features=256, n_heads=4, n_attn_modules=2, n_linears=4):
        """
        Parameters
        ----------
        in_channels: int (default 1)
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
        super(BoxWorldNet, self).__init__()
        
        self.n_features = n_features
        
        MLP = clones( nn.Linear(n_features,n_features), n_linears)
        
        self.net = nn.Sequential(
            ExtractEntities(n_kernels, in_channels),
            RelationalModule(n_kernels, n_features, n_heads, n_attn_modules),
            FeaturewiseMaxPool(pixel_axis = 0),
            *MLP)
        
        if debug:
            print(self.net)
        
    def forward(self, x):
        x = self.net(x)
        if debug:
            print("x.shape (BoxWorldNet): ", x.shape)
        return x
           
        
        
        
        
        
        
        
        
