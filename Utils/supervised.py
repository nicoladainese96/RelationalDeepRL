import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from Utils import test_env

def play_optimal(env):
    """
    Play optimal policy with maximum entropy (equal probability allocation to all optimal choices).
    """
    state = env.reset(random_init = False)
    
    actions = []
    states = []

    while True:
        action = env.get_optimal_action()
        actions.append(action)
        
        new_state, reward, terminal, info = env.step(action) 
        states.append(new_state)
        
        if terminal:
            break
            
        state = new_state
    
    return actions, states

def random_start(X=10, Y=10):
    s1, s2 = np.random.choice(X*Y, 2, replace=False)
    initial = [s1//X, s1%X]
    goal = [s2//X, s2%X]
    return initial, goal

def create_action_state_set(game_params, size = 10000):
    action_memory = []
    state_memory = []
    
    while len(action_memory) < size:
        
        # Change game params
        initial, goal = random_start(game_params["x"], game_params["y"])

        # All game parameters
        game_params["initial"] = initial
        game_params["goal"] = goal

        env = test_env.Sandbox(**game_params)
        
        actions, states = play_optimal(env)
        action_memory += actions
        state_memory += states
        
        #print('len(action_memory): ',len(action_memory))
        
    return np.array(state_memory[:size]), np.array(action_memory[:size])

class NumpyDataset(Dataset):
    """
    Class to interface numpy dataset with torch DataLoader
    """
    
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return (self.data[index], self.label[index])
    
def prepare_dataset(x, y, train_perc, val_perc, train_batch_size=128, val_batch_size=128, test_batch_size=128):
    """
    Parameters
    ----------
    x: numpy array
        Shape (samples, other dims) - dataset
    y: numpy array
        Shape (samples) - labels
    train_perc: float in (0,1)
        Percentage of dataset to be used during training (validation set included)
    val_per: float in (0,1)
        Percentage of the training set to be used as validation set
    train_batch_size, val_batch_size, test_batch_size: int
        Training, validation and test batch sizes. Default 128
        
    Returns
    -------
    train_loader, val_loader, test_loader: torch DataLoader instances
    """
    
    # training/test splitting
    m = int(len(x)*train_perc)
    x_train= x[:m]
    y_train = y[:m]
    x_test =  x[m:]
    y_test = y[m:]
    
    # define custom NumpyDatasets
    train_set = NumpyDataset(x_train, y_train)
    test_set =  NumpyDataset(x_test, y_test)
   
    train_len = int(m*(1-val_perc))
    train_sampler = SubsetRandomSampler(np.arange(train_len))
    val_sampler = SubsetRandomSampler(np.arange(train_len,m))

    train_loader = DataLoader(train_set, train_batch_size, sampler=train_sampler, drop_last=True, collate_fn=lambda x: x)
    val_loader = DataLoader(train_set, val_batch_size, sampler=val_sampler, drop_last=True, collate_fn=lambda x: x)
    test_loader = DataLoader(test_set, test_batch_size, drop_last=False, collate_fn=lambda x: x)

    return train_loader, val_loader, test_loader

def test_epoch(net, dataloader, loss_fn, optimizer):

    # select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # Validation
    net.eval() # Evaluation mode (e.g. disable dropout)
    with torch.no_grad(): # No need to track the gradients
        batch_len = np.zeros(len(dataloader))
        batch_loss = np.zeros(len(dataloader))
        for i, data in enumerate(dataloader,0):
            # Extract data and move tensors to the selected device
            x = [x[0] for x in data]
            x = torch.tensor(x).float().to(device)
            
            y =  [x[1] for x in data]
            y = torch.LongTensor(y).to(device)

            y_pred = net(x)

            loss = loss_fn(y_pred, y)
            
            # save MSE loss and length of a batch
            batch_len[i] = len(data)
            batch_loss[i] = loss.item()
    
    # total loss
    val_loss = (batch_loss*batch_len).sum()/batch_len.sum()
    return val_loss

def train_NN(net, lr, n_epochs, train_loader, val_loader, train_log=True, verbose=True, 
                  debug=False, return_model = False):
    """
    Trains a Pytorch network.
    
    Parameters
    ----------
    model: Pytorch nn.Module instance
        Must have forward method
    lr: float
        Learning rate
    n_epochs: int
        Number of epochs of training
    train_loader: torch DataLoader
        Loads the training set
    val_loader: torch DataLoader
        Loads the validation set
    verbose: bool
        If True prints updates of the training 10 times for each epoch
    return_model: bool
        If True returns the trained instance of the model 
    **params: dictionary 
        Must contain all the parameters needed by the model, the optimizer and the loss
    
    Returns
    -------
    net (if return_model): Pytorch nn.Module class
        Trained instance of the model 
    train_loss_log (if train_log): list
        Training loss for each epoch
    val_loss_log (if train_log): list
        Validation loss for each epoch
    val_acc_log (if train_log): list
        Validation accuracy for each epoch
    
    """
  
    optimizer = optim.Adamax(net.parameters(), lr)
    loss_fn = nn.NLLLoss()
    
    # define contextual print functions activated by print flags
    verbose_print = print if verbose else lambda *a, **k: None
    verbose_print("Verbose: ", verbose)
    dprint = print if debug else lambda *a, **k: None
    dprint("Debug: ", debug)

    # If cuda is available set the device to GPU
    verbose_print("Using cuda: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Move all the network parameters to the selected device (if they are already on that device nothing happens)
    net.to(device)
    
    n_batches = len(train_loader)
    epoch_time = []
    #Time for printing
    training_start_time = time.time()
    # lists with the history of the training
    if (train_log == True):
        train_loss_log = []
        val_loss_log = []

    #Loop for n_epochs
    for epoch in range(n_epochs):

        running_loss = 0.0
        print_every = n_batches // 10 # frequency of printing
        start_time = time.time()
        total_train_loss = 0
        batches_done = 0
        net.train() # activate dropout
        for i, data in enumerate(train_loader, 0):
            batches_done += 1
            
            x = [x[0] for x in data]
            x = torch.tensor(x).float().to(device)
            
            y =  [x[1] for x in data]
            y = torch.LongTensor(y).to(device)

            y_pred = net(x)

            loss = loss_fn(y_pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Print statistics
            running_loss += loss.item() 
            total_train_loss += loss.item()
            #Print every 10th batch of an epoch
            if ((i+1) % (print_every) == 0) or (i == n_batches - 1):
                verbose_print('\r'+"Epoch {}, {:d}% \t Train loss: {:.4f} took: {:.2f}s ".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / batches_done,
                        time.time() - start_time), end=' ')
                
        epoch_time.append(time.time() - start_time)
        if (train_log == True):
            train_loss_log.append(total_train_loss/len(train_loader))
        
        
        #At the end of the epoch, do a pass on the validation set
        val_loss = test_epoch(net, dataloader=val_loader, loss_fn=loss_fn, optimizer=optimizer) 
        if (train_log == True):
            val_loss_log.append(val_loss)
            verbose_print("Val. loss: {:.4f}".format(val_loss ))

    verbose_print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    if train_log:
        if return_model:
            return net, train_loss_log, val_loss_log#, val_acc_log
        else:
            return train_loss_log, val_loss_log#, val_acc_log  #used during cross validation
        
        
def plot_results(train_loss, val_loss):
    n_epochs = np.arange(1,len(train_loss)+1)
    plt.figure(figsize=(8,6))
    plt.plot(n_epochs, train_loss, label='train')
    plt.plot(n_epochs, val_loss, label='val')
    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(fontsize=13)
    plt.show()
    
    
class OheAgent():
    def __init__(self, agent_net, device):
        self.net = agent_net
        self.device = device
        self.net.to(device)
        
    def get_action(self, state, return_log=True):
        state = torch.tensor(state).float().to(self.device)
        log_probs = self.net(state)
        dist = torch.exp(log_probs)
        probs = Categorical(dist)
        action =  probs.sample().item()
        if return_log:
            return action, log_probs.view(-1)[action], dist
        else:
            return action
