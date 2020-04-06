import sys

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
        
def get_state(observation, mask=True):
    #print("Keys: ", observation.layers.keys())
    board = copy.deepcopy(observation.board)#.astype('float')
    if debug: print("mask: ", mask)
    if mask:
        walls = observation.layers['#'].astype(int)
        #print("walls: ", walls)
        background = observation.layers[' '].astype(int)
        #print("background: ", background)
        ambient = walls + background
        #print("ambient: ", ambient)
        board[ambient.astype(bool)] = 0
        #print("board (masked): ", board)
    grid_size = board.shape[0]
    board = board.reshape(1, grid_size, grid_size)
    return board #/MAX_PIXEL

def play_episode(agent, game, max_steps, mask=True):

    # Start the episode
    observation, _, _ = game.its_showtime()
    state = get_state(observation, mask)
    
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
        new_state = get_state(new_obs, mask)
        
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

    return rewards, log_probs, distributions, np.array(states), done, bootstrap

def train_boxworld(agent, game_params, n_episodes = 1000, max_steps=120, return_agent=False, mask=True):
    performance = []
    time_profile = []
    
    for e in range(n_episodes):
        
        #print("Playing episode %d... "%(e+1))
        t0 = time.time()
        game = bw.make_game(**game_params)
        rewards, log_probs, distributions, states, done, bootstrap = play_episode(agent, game, max_steps, mask)
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

