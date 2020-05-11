import numpy as np
import torch
from Utils import test_env
import time

debug = False

def play_episode(agent, env, max_steps):

    # Start the episode
    state = env.reset()
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
    done = np.array(done)
    bootstrap = np.array(bootstrap)

    return rewards, log_probs, distributions, np.array(states), done, bootstrap

def random_start(X=10, Y=10):
    s1, s2 = np.random.choice(X*Y, 2, replace=False)
    initial = [s1//X, s1%X]
    goal = [s2//X, s2%X]
    return initial, goal

def train_sandbox(agent, game_params, lr, n_episodes = 1000, max_steps=120, return_agent=False, random_init=True):
    performance = []
    steps_to_solve = []
    time_profile = []
    critic_losses = [] 
    actor_losses = []
    entropies = []
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    
    for e in range(n_episodes):
        
        if random_init:
            # Change game params
            initial, goal = random_start(game_params["x"], game_params["y"])

            # All game parameters
            game_params["initial"] = initial
            game_params["goal"] = goal

        #print("Playing episode %d... "%(e+1))
        t0 = time.time()
        env = test_env.Sandbox(**game_params)
        rewards, log_probs, distributions, states, done, bootstrap = play_episode(agent, env, max_steps)
        t1 = time.time()
        #print("Time playing the episode: %.2f s"%(t1-t0))
        performance.append(np.sum(rewards))
        steps_to_solve.append(len(rewards))
        if (e+1)%10 == 0:
            print("Episode %d - reward: %.2f - steps to solve: %.2f"%(e+1, np.mean(performance[-10:]), np.mean(steps_to_solve[-10:])))
        #print("Episode %d - reward: %.2f - steps to solve: %d"%(e+1, performance[-1], len(rewards)))

        critic_loss, actor_loss, entropy = agent.compute_ac_loss(rewards, log_probs, distributions, states, done, bootstrap)
        loss = (critic_loss+actor_loss).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        agent.update_target()
        
        critic_losses.append(critic_loss.item())
        actor_losses.append(actor_loss.item())
        entropies.append(entropy.item())
    
        t2 = time.time()
        #print("Time updating the agent: %.2f s"%(t2-t1))
            
        time_profile.append([t1-t0, t2-t1])
        
    performance = np.array(performance)
    time_profile = np.array(time_profile)
    steps_to_solve = np.array(steps_to_solve)
    L = n_episodes // 6 # consider last sixth of episodes to compute agent's asymptotic performance
    losses = dict(critic_losses=critic_losses, actor_losses=actor_losses, entropies=entropies)
    if return_agent:
        return performance, performance[-L:].mean(), performance[-L:].std(), agent, time_profile, losses, steps_to_solve
    else:
        return performance, performance[-L:].mean(), performance[-L:].std(), losses, steps_to_solve
