import numpy as np
import torch
from RelationalModule import ActorCritic
from Utils import test_env
import time
from importlib import reload

debug = False

def play_episode(agent, env, max_steps):

    # Start the episode
    state = env.reset()
    if debug: print("state.shape: ", state.shape)
    rewards = []
    log_probs = []
    distributions = []
    states = [state]
    done = []
    bootstrap = []
        
    steps = 0
    while True:
     
        action, log_prob, distrib = agent.get_action(state, return_log = True)
        new_state, reward, terminal, info = env.step(action)
        if debug: print("state.shape: ", new_state.shape)
        rewards.append(reward)
        log_probs.append(log_prob)
        distributions.append(distrib)
        states.append(new_state)
        done.append(terminal)
        
        # Still unclear how to retrieve max steps from the game itself
        if terminal is True and steps == max_steps:
            bootstrap.append(True)
        else:
            bootstrap.append(False) 
        
        if terminal is True:
            #print("steps: ", steps)
            #print("Bootstrap needed: ", bootstrap[-1])
            break
            
        state = new_state
        steps += 1
        
    rewards = np.array(rewards)
    states = np.array(states)
    if debug: print("states.shape: ", states.shape)
    done = np.array(done)
    bootstrap = np.array(bootstrap)

    return rewards, log_probs, distributions, np.array(states), done, bootstrap

def train_sandbox(agent, game_params, n_episodes = 1000, max_steps=120, return_agent=False):
    performance = []
    time_profile = []
    
    for e in range(n_episodes):
        
        #print("Playing episode %d... "%(e+1))
        t0 = time.time()
        env = test_env.Sandbox(**game_params)
        rewards, log_probs, distributions, states, done, bootstrap = play_episode(agent, env, max_steps)
        t1 = time.time()
        #print("Time playing the episode: %.2f s"%(t1-t0))
        performance.append(np.sum(rewards))
        if (e+1)%10 == 0:
            print("Episode %d - reward: %.2f"%(e+1, np.mean(performance[-10:])))
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
