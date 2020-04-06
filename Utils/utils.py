import string
import random
import os
import numpy as np 
import sys
sys.path.insert(0, "pycolab/pycolab/examples/research/box_world")
import box_world as bw
import matplotlib.pyplot as plt
import time

def load_session(load_dir, keywords):
    filenames = os.listdir(load_dir)
    matching_filenames = []
    for f in filenames:
        if np.all([k in f.split('_') for k in keywords]):
            matching_filenames.append(f)

    print("Number of matching filenames: %d\n"%len(matching_filenames), matching_filenames)
    

    matching_dicts = []
    for f in matching_filenames:
        d = np.load(load_dir+f, allow_pickle=True)
        matching_dicts.append(d)

    if len(matching_dicts) == 1:
        return matching_dicts[0].item()
    else:
        return matching_dicts

def save_session(save_dir, keywords, game_params, HPs, score):
    ID = ''.join([random.choice(string.ascii_letters) for _ in range(4)])
    ID = ID.upper()
    keywords.append(ID)
    filename = '_'.join(keywords)
    filename = 'S_'+filename
    print("Save at "+save_dir+filename)
    train_session_dict = dict(game_params=game_params, HPs=HPs, score=score, n_epochs=len(score), keywords=keywords)
    np.save(save_dir+filename, train_session_dict)
    return ID

def rendering(agent, game_params, save=False, max_steps=120):
    fig = plt.figure(figsize = (8,6))
    
    # Make environment
    game = bw.make_game(**game_params)
    
    # Start the episode
    observation, _, _ = game.its_showtime()
    state = get_state(observation)
    
    ### Rendering ###
    rgb_map = grey_to_RGB(observation.board)
    
    plt.imshow(rgb_map) # show map
    plt.title("BoxWorld Env - Turn: %d"%(0))
    plt.yticks([])
    plt.xticks([])
    fig.show()
    time.sleep(0.75) #uncomment to slow down for visualization purposes
    if save:
        plt.savefig('.raw_gif/turn%.3d.png'%0)
            
    steps = 0
    while True:
        
        action, log_prob = agent.get_action(state, return_log = True)
        new_obs, reward, not_terminal = game.play(action)
        not_terminal = bool(not_terminal)
        new_state = get_state(new_obs)
        
         ### Rendering ###
            
        rgb_map = grey_to_RGB(new_obs.board)
        plt.cla() # clear current axis from previous drawings -> prevents matplotlib from slowing down
        plt.imshow(rgb_map)
        plt.title("Boxworld Env - Turn: %d "%(steps+1))
        plt.yticks([]) # remove y ticks
        plt.xticks([]) # remove x ticks
        fig.canvas.draw() # update the figure
        time.sleep(0.75) #uncomment to slow down for visualization purposes
        if save:
            plt.savefig('.raw_gif/turn%.3d.png'%(steps+1))
            
        if not_terminal is False:
            break
            
        state = new_state
        steps += 1

def grey_to_RGB(grey_map, seeds=[100,200,300]):
    
    def random_int_mapping(vocab_size = 256, seed=None):
        np.random.seed(seed)
        mapping = np.random.permutation(np.arange(vocab_size))
        np.random.seed(None)
        return mapping

    mappings = [random_int_mapping(seed=s) for s in seeds]
    shape = grey_map.shape
    RGB_img = []
    
    for i in range(len(mappings)):
        colormap = mappings[i][grey_map.flatten()].reshape(shape)
        RGB_img.append(colormap)
        
    return np.stack(RGB_img, axis=2)

def get_state(observation):
    board = observation.board
    grid_size = board.shape[0]
    board = board.reshape(1, grid_size, grid_size)
    return board 
