import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from Utils import test_env

def supervised_training(net, lr, n_epochs, n_samples, game_params, get_probs=False):
    env = test_env.Sandbox(**game_params)
    print("\nCreating dataset...")
    state_set, action_set = create_action_state_set(game_params, size=n_samples, get_probs=get_probs)
    train_loader, val_loader, test_loader = prepare_dataset(state_set, action_set, 0.8, 0.2)
    dataloader_dict = dict(train_loader=train_loader,
                           val_loader=val_loader,
                           test_loader=test_loader)
    print("\nTraining network...")
    net, train_loss, val_loss = train_NN(net, lr, n_epochs, train_loader, val_loader, 
                                         return_model=True, KL_loss=get_probs)
    
    return net, train_loss, val_loss, dataloader_dict, state_set, action_set, env

def play_optimal(env, get_probs=False):
    """
    Play optimal policy with maximum entropy (equal probability allocation to all optimal choices).
    """
    state = env.reset(random_init = False)
    
    if get_probs:
        probabilities = []
    else:
        actions = []
    states = []

    while True:
        if get_probs:
            action, probs = env.get_optimal_action(show_all=get_probs)
            probabilities.append(probs)
        else:
            action = env.get_optimal_action()
            actions.append(action)
        
        new_state, reward, terminal, info = env.step(action) 
        states.append(new_state)
        
        if terminal:
            break
            
        state = new_state
    
    if get_probs:
        return probabilities, states
    else:
        return actions, states

def random_start(X=10, Y=10):
    s1, s2 = np.random.choice(X*Y, 2, replace=False)
    initial = [s1//X, s1%X]
    goal = [s2//X, s2%X]
    return initial, goal

def create_action_state_set(game_params, size = 10000, get_probs=False):
    action_memory = []
    state_memory = []
    
    while len(action_memory) < size:
        
        # Change game params
        initial, goal = random_start(game_params["x"], game_params["y"])

        # All game parameters
        game_params["initial"] = initial
        game_params["goal"] = goal

        env = test_env.Sandbox(**game_params)
        
        actions, states = play_optimal(env, get_probs)
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

def test_epoch(net, dataloader, loss_fn, optimizer, KL_loss):

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
            
            if KL_loss:
                y = torch.tensor(y).float().to(device)
            else:
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
                  debug=False, return_model = False, KL_loss=False):
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
  
    optimizer = optim.Adamax(net.parameters(), lr, weight_decay=1e-5)
    if KL_loss:
        loss_fn = nn.KLDivLoss()
    else:
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
            if KL_loss:
                y = torch.tensor(y).float().to(device)
            else:
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
        val_loss = test_epoch(net, dataloader=val_loader, loss_fn=loss_fn, optimizer=optimizer, KL_loss=KL_loss) 
        if (train_log == True):
            val_loss_log.append(val_loss)
            verbose_print("Val. loss: {:.4f}".format(val_loss ))

    verbose_print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    if train_log:
        if return_model:
            return net, train_loss_log, val_loss_log#, val_acc_log
        else:
            return train_loss_log, val_loss_log#, val_acc_log  #used during cross validation
        
def plot_decision_map(env, net, goal, coord=True):
    env.goal = goal
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    probabilities = []
    for x in range(env.boundary[0]):
        for y in range(env.boundary[1]):
            if [x,y] != goal:
                env.state = [x,y]
                if coord:
                    enc_state = env.enc_to_coord()
                else:
                    grey_state = env.enc_to_grey()
                    enc_state = env.grey_to_onehot(grey_state)
                enc_state = torch.tensor(enc_state).float().to(device).unsqueeze(0)
                #print("enc_state.shape: ", enc_state.shape)
                log_probs = net(enc_state).squeeze()
                probs = torch.exp(log_probs).cpu().detach().numpy()
                
            else:
                probs = np.zeros(env.n_actions)
            probabilities.append(probs)
            
    probs = np.array(probabilities).reshape((env.boundary[0],env.boundary[1],-1))
    
    fig = plt.figure(figsize=(8,8))
    fig.suptitle('Goal in [%d,%d]'%(goal[0],goal[1]), fontsize=18, y=1, x=0.48)
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)
    ax3 = plt.subplot(2,2,3)
    ax4 = plt.subplot(2,2,4)
    axes = [ax1, ax2, ax3, ax4]
    for i, ax in enumerate(axes):
        im = ax.imshow(probs[:,:,i], cmap='plasma', vmin=0, vmax=1)
        ax.set_title("Prob of moving "+env.action_dict[i], fontsize=16)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        fig.colorbar(im, ax=ax, shrink=0.77)
    plt.tight_layout()
    
    return probs

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
