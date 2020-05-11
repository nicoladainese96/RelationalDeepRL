import numpy as np
import torch
import torch.multiprocessing as mp
import time
import copy 
from AC_modules.Networks import *
from AC_modules.AdvantageActorCritic import SharedAC, IndependentAC

from Utils import test_env

debug = True
queue = True

def build_AC(model_dict): # works
    if model_dict['shared']:
        return SharedAC(model_dict['model'], *model_dict['args'], **model_dict['kwargs'])
    else:
        return IndependentAC(model_dict['model'], *model_dict['args'], **model_dict['kwargs'])
        
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

def training_thread(global_model, model_constructor, game_params, learning_rate, n_episodes, 
                    max_steps, random_init, rank, optim_steps, env_steps):
    if debug:
        print("Entered process %d"%rank)
        print("Setting numpy and torch seeds...")
    np.random.seed()
    seed = np.random.randint(10000)
    torch.manual_seed(seed)
    #local_model = model_constructor.generate_model()
    #print("Constructed local model ")
    #print("model_dict: ", model_dict)
    local_model = model_constructor.generate_model()
    if debug: print("Constructed local model ")
        

    local_model.load_state_dict(global_model.state_dict())
    if debug: print("Loaded state dictionary")
    
    optimizer = torch.optim.Adam(global_model.parameters(), lr=learning_rate)
    if debug: print("Created optim")
    
    print("Process %d started"%rank)
    
    performance = []
    steps_to_solve = []
    for e in range(n_episodes):
        if debug: print("Episode %d - process %d"%(e+1, rank))
        if random_init:
            # Change game params
            initial, goal = random_start(game_params["x"], game_params["y"])

            # All game parameters
            game_params["initial"] = initial
            game_params["goal"] = goal

        env = test_env.Sandbox(**game_params)
        if debug: print("Environemnt created")
        rewards, log_probs, distributions, states, done, bootstrap = play_episode(local_model, env, max_steps)
        if debug: print("Episode played")
        env_steps.value += len(rewards) # hope it works asynchronously - TO CHECK
        
        performance.append(np.sum(rewards))
        steps_to_solve.append(len(rewards))
        
        #if (e+1)%10 == 0:
        #    print("Episode %d of process %d - reward: %.2f - steps to solve: %.2f"%(e+1, rank, np.mean(performance[-10:]), np.mean(steps_to_solve[-10:])))
        #print("Episode %d of process %d - reward: %.2f - steps to solve: %.2f"%(e+1, rank, performance[-1], steps_to_solve[-1]))
        
        critic_loss, actor_loss, entropy = local_model.compute_ac_loss(rewards, log_probs, distributions, states, done, bootstrap)
        #print("Loss computed")
        loss = (critic_loss + actor_loss).mean()
        #print("Loss summed")
        # Update global model and then copy back updated params to local model
        optimizer.zero_grad()
        #print("Gradient zeroed")
        loss.backward()
        #print("Backward")
        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            global_param._grad = local_param.grad
        #print("Gradient copied")
        optimizer.step()
        optim_steps.value += 1 # hope it works asynchronously - TO CHECK
        #global_model.update_target()
        #if debug: print("Updated target")
        local_model.load_state_dict(global_model.state_dict())
        if debug: print("Loaded state dictionary")
            
    print("Training process {} reached maximum episode.".format(rank))
    
    
def test_thread(global_model, game_params, tot_episodes, max_steps, random_init, optim_steps, n_processes,
                Q=None, episodes_per_test=1, test_every=10):
    print("Test process started")
    test_counter = 1
    max_tests = (tot_episodes*n_processes // test_every) -n_processes
    print("max_tests: ", max_tests)
    
    np.random.seed()
    seed = np.random.randint(10000)
    torch.manual_seed(seed)
    
    performance = []
    steps_to_solve = []
    critic_losses = [] 
    actor_losses = []
    entropies = []
    optimizer_steps = []
    
    while True:
        print("optim_steps: ", optim_steps.value)
        print("test counter: ", test_counter)
        if optim_steps.value > test_counter*test_every:
            test_counter +=1
            episode_reward = []
            episode_steps = []
            for e in range(episodes_per_test):
                if random_init:
                    # Change game params
                    initial, goal = random_start(game_params["x"], game_params["y"])

                    # All game parameters
                    game_params["initial"] = initial
                    game_params["goal"] = goal

                env = test_env.Sandbox(**game_params)
                rewards, log_probs, distributions, states, done, bootstrap = play_episode(global_model, env, max_steps)
                episode_reward.append(np.sum(rewards))
                episode_steps.append(len(rewards))
                
                if e == 0:
                    critic_loss, actor_loss, entropy = global_model.compute_ac_loss(rewards, log_probs, distributions, states, done, bootstrap)
            performance.append(np.mean(episode_reward))
            steps_to_solve.append(np.mean(episode_steps))

            if test_counter % 10:
                print("Test %d - reward %.2f - steps to solve %.2f"%(test_counter, np.mean(performance[-10:]), np.mean(steps_to_solve[-10:])))
            critic_losses.append(critic_loss.detach().cpu().item())
            actor_losses.append(actor_loss.detach().cpu().item())
            entropies.append(entropy.detach().cpu().item())

        else:
            time.sleep(0.01) # wait 1 sec
            #pass
        if test_counter == max_tests:
            break
    if queue:
        print("Putting results in queue")
        #Q.put([performance, steps_to_solve, critic_losses, actor_losses, entropies]) # TO CHECK  
        print("performance (put) ")#, performance)
        Q.put(performance)
        #time.sleep(0.5)
        print("steps_to_solve (put)")#, steps_to_solve)
        Q.put(steps_to_solve)
        #time.sleep(0.5)
        print("critic_losses (put)")#, critic_losses)
        Q.put(critic_losses)
        #time.sleep(0.5)
        print("actor_losses (put) ")#, actor_losses)
        Q.put(actor_losses)
        #time.sleep(0.5)
        print("entropies (put)")#, entropies)
        Q.put(entropies)
        print("optimizer_steps (put) ")#, optimizer_steps)
        Q.put(optimizer_steps)
        time.sleep(0.1)

        
def train_sandbox(agent_constructor, learning_rate, game_params, n_training_threads=3, n_episodes=1000,
                  max_steps=120, return_agent=False, random_init=True):
    
    global_model = agent_constructor.generate_model()
    #global_model.init_target()
    global_model.share_memory()
    optim_steps = mp.Value('i')
    env_steps = mp.Value('i')
    if queue:
        Q = mp.Queue() # TO CHECK
    processes = []
    for rank in range( n_training_threads + 1):  # + 1 for test process
        if rank == 0:
            if queue:
                p = mp.Process(target=test_thread, args=(global_model, game_params, n_episodes, 
                                                     max_steps, random_init, optim_steps, n_training_threads, Q,))
            else:
                p = mp.Process(target=test_thread, args=(global_model, game_params, n_episodes, 
                                                     max_steps, random_init, optim_steps, n_training_threads,))
        else:
            p = mp.Process(target=training_thread, args=(global_model, agent_constructor, game_params, learning_rate, n_episodes,
                                                         max_steps, random_init, rank, optim_steps, env_steps,))
        p.start()
        processes.append(p)
    print("All processes started")
    
    if queue:
        performance = Q.get() # TO CHECK
        print("performance (get)")#, performance)

        steps_to_solve = Q.get() # TO CHECK
        print("steps_to_solve (get)")#, steps_to_solve)
        critic_losses = Q.get() # TO CHECK
        print("critic_losses (get)")#, critic_losses)
        actor_losses = Q.get() # TO CHECK
        print("actor_losses (get)")#, actor_losses)
        entropies = Q.get() # TO CHECK
        print("entropies (get)")#, entropies)
        optimizer_steps = Q.get()
        print("optimizer_steps (get)")#, optimizer_steps)
        Q.close()
        #performance, steps_to_solve, critic_losses, actor_losses, entropies = results
        losses = dict(critic_losses=critic_losses, actor_losses=actor_losses, entropies=entropies)

        
    for p in processes:
        p.join()
    print("All processes finished")
        
    if queue:
        if return_agent:
            return performance, global_model, losses, steps_to_solve, optimizer_steps
        else:
            return performance, losses, steps_to_solve, optimizer_steps
    else:
        if return_agent:
            return global_model
