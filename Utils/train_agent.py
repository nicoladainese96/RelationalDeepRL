import sys

sys.path.insert(0, "RelationalDeepRL/pycolab/pycolab/examples/research/box_world")
sys.path.insert(0, "pycolab/pycolab/examples/research/box_world")

import box_world as bw

import numpy as np
import torch
from RelationalModule import ActorCritic
import time
from importlib import reload
reload(ActorCritic)
import copy

MAX_PIXEL = 116
show = False
debug = False

def show_game_state(observation):
    for row in observation.board: print(row.tostring().decode('ascii'))
        
def get_color_dict(seed=100):
    import string
    alphabet = string.ascii_lowercase
    np.random.seed(seed)
    RGB_list = np.random.rand(len(alphabet),3)
    color_dict = {}
    for l, c in zip(alphabet,RGB_list):
        color_dict[l] = c
    np.random.seed(None)
    
    # add colors for 'agent' and  'gem'
    color_dict['agent'] = np.array([1.,0.,0.])
    color_dict['gem'] = np.array([1.,1.,1.])
    
    return color_dict

def get_state(observation, state_dictionaries):
    
    color_dict, object_dict, symbol_dict = state_dictionaries 
    b = observation.board
    l = observation.layers  
    color_board = np.zeros(b.shape+(3,)).astype(float)
    object_board = np.zeros(b.shape+(1,)).astype(int)

    for symbol in l.keys():

        # If alphabetic character
        if symbol.isalpha():
            # Paint the color board cells occupied accordingly
            color_board[l[symbol]] = color_dict[symbol.lower()]

            # Upper = box, lower = key
            if symbol.isupper():
                object_board[l[symbol]] = object_dict['box']
            else:
                object_board[l[symbol]] = object_dict['key']

        else:
            object_name = symbol_dict[symbol]

            # Color assigned is [0,0,0] since it's not really a property of those objects
            # Only agent and gem have colors mainly for plotting reasons
            if object_name == 'agent':
                color_board[l[symbol]] = color_dict['agent']
            elif object_name == 'gem':
                color_board[l[symbol]] = color_dict['gem']
            else:
                pass

            object_board[l[symbol]] = object_dict[object_name]
            
    return (object_board, color_board)
        

def play_episode(agent, game, max_steps):
    
    # Define dictionaries to represent state from observation
    color_dict = get_color_dict()
    object_dict = {'ground':0, 'wall':1, 'agent':2, 'gem':3, 'key':4, 'box':5}
    symbol_dict = {' ':'ground', '#':'wall', '.':'agent', '*':'gem'} # keys and boxes can have any possible letter
    state_dictionaries = [color_dict, object_dict, symbol_dict]
    
    # Start the episode
    observation, _, _ = game.its_showtime()
    state = get_state(observation, state_dictionaries)
    
    rewards = []
    log_probs = []
    distributions = []
    states = [state]
    not_done = []
    bootstrap = []

    steps = 0
    while True:
     
        action, log_prob, distrib = agent.get_action(state, return_log = True)
        new_obs, reward, not_terminal = game.play(action)
        not_terminal = bool(not_terminal)

        if show:
            show_game_state(new_obs)
        new_state = get_state(new_obs, state_dictionaries)
        
        rewards.append(reward)
        log_probs.append(log_prob)
        distributions.append(distrib)
        states.append(new_state)
        not_done.append(not_terminal)
        
        # Still unclear how to retrieve max steps from the game itself
        if not_terminal is False and steps == max_steps:
            bootstrap.append(True)
        else:
            bootstrap.append(False) 
        
        if not_terminal is False:
            #print("steps: ", steps)
            #print("Bootstrap needed: ", bootstrap[-1])
            break
            
        state = new_state
        steps += 1
        
    rewards = np.array(rewards)
    done = ~np.array(not_done)
    bootstrap = np.array(bootstrap)

    return rewards, log_probs, distributions, states, done, bootstrap

def train_boxworld(agent, game_params, n_episodes = 1000, max_steps=120, return_agent=False):
    performance = []
    time_profile = []
    
    for e in range(n_episodes):
        
        #print("Playing episode %d... "%(e+1))
        t0 = time.time()
        game = bw.make_game(**game_params)
        rewards, log_probs, distributions, states, done, bootstrap = play_episode(agent, game, max_steps)
        t1 = time.time()
        #print("Time playing the episode: %.2f s"%(t1-t0))
        performance.append(np.sum(rewards))
        if (e+1)%100 == 0:
            print("Episode %d - reward: %.2f"%(e+1, np.mean(performance[-100:])))
        #print("Episode %d - reward: %.0f"%(e+1, performance[-1]))

        agent.update(rewards, log_probs, distributions, states, done, bootstrap)
        t2 = time.time()
        #print("Time updating the agent: %.2f s"%(t2-t1))
            
        time_profile.append([t1-t0, t2-t1])
        
    performance = np.array(performance)
    time_profile = np.array(time_profile)
    L = n_episodes // 6 # consider last sixth of episodes to compute agent's asymptotic performance
    
    if return_agent:
        return performance, performance[-L:].mean(), performance[-L:].std(), agent, time_profile
    else:
        return performance, performance[-L:].mean(), performance[-L:].std()

