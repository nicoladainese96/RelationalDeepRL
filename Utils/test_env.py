import numpy as np
import time

# same values as in BoxWorld
AGENT_COLOR = 46
GOAL_COLOR = 42
BACKGROUND_COLOR = 32

debug = False

class Sandbox():
    
    def __init__(self, x, y, initial, goal, R0=0, max_steps=0, greyscale_state=True):
        
        self.greyscale = greyscale_state
        self.boundary = np.asarray([x, y])
        self.initial = np.asarray(initial)
        self.state = np.asarray(initial)
        self.goal = goal
        self.R0 = R0
        # top-left corner is [0,0], bottom-right is [x,y]
        # vertical direction is x (first coordinate)
        # horizontal direction is y (second coordinate)
        self.action_map = {
                            0: [0, 1],
                            1: [0, -1],
                            2: [1, 0],
                            3: [-1, 0],
                          }
        self.action_dict = {
                            0: 'Right',
                            1: 'Left',
                            2: 'Down',
                            3: 'Up',
                          }
        self.n_actions = len(self.action_map.keys())
        if max_steps == 0:
            self.max_steps = 5*int(np.max([x,y])) # 5 times the greatest linear dimension
        else:
            self.max_steps = max_steps
        self.current_steps = 0
        
    def step(self, action):
        
        self.current_steps += 1
        
        # Baseline reward
        reward = self.R0 
        
        # Get grid movement
        movement = self.action_map[action]
        
        # Compute next vectorial state
        next_state = self.state + np.asarray(movement)
        
        if not (self.check_boundaries(next_state)):
            # Enforce staying within boundaries with negative reward
            reward = -1
        else:
            # Update state only if valid movement
            self.state = next_state
            
        if (self.state == self.goal).all():
            reward = 1
            terminal = True
        else:
            terminal = False
        
        # Check if number of steps has exceeded the maximum for an episode
        info = {}
        if self.current_steps == self.max_steps:
            terminal = True
            info['TimeLimit.truncated'] = True
            
        if self.greyscale:
            enc_state = self.enc_to_grey()
        else:
            enc_state = self.encode_state()
            
        return enc_state, reward, terminal, info

    def check_boundaries(self, state):
        x_ok = (state[0] >= 0) and (state[0] < self.boundary[0])
        y_ok = (state[1] >= 0) and (state[1] < self.boundary[1])
        
        if x_ok and y_ok:
            return True
        else:
            return False
        
    def encode_state(self):
        # encode row by row
        # e = X*y + x
        enc_state = self.boundary[0]*self.state[1] + self.state[0]
        return enc_state
    
    def enc_to_grey(self):
        grey_img = np.full(self.boundary, BACKGROUND_COLOR)
        grey_img[self.state[0],self.state[1]] = AGENT_COLOR
        grey_img[self.goal[0],self.goal[1]] = GOAL_COLOR
        return np.array([grey_img])
    
    def reset(self, random_init=False):
        if random_init:
            self.initial[0] = np.random.choice(self.boundary[0]-1)
            self.initial[1] = np.random.choice(self.boundary[1]-1)
        self.state = self.initial
        self.current_steps = 0
        
        if self.greyscale:
            return self.enc_to_grey()
        else:
            return self.encode_state()
    
    def dist_to_goal(self, state):
        dx = np.abs(state[0] - self.goal[0])
        dy = np.abs(state[1] - self.goal[1])
        return dx + dy
    
    def get_optimal_action(self):
        optimal = np.zeros(self.n_actions)
        d0 = self.dist_to_goal(self.state)
        # consider all actions
        for action in range(self.n_actions):
            # compute for each the resulting state)
            movement = self.action_map[action]
            next_state = self.state + np.asarray(movement)
            if(self.check_boundaries(next_state)):
                # if the state is admitted -> compute the distance to the goal 
                d = self.dist_to_goal(next_state)
                # if the new distance is smaller than the old one, is an optimal action (optimal = 1.)
                if d < d0:
                    optimal[action] = 1.
                else:
                    optimal[action] = 0.
            else:
                # oterwise is not (optimal = 0)
                optimal[action] = 0.
        # once we have the vector of optimal, divide them by the sum
        probs = optimal/optimal.sum()
        # finally sample the action and return it together with the log of the probability
        opt_action = np.random.choice(self.n_actions, p=probs)
        return opt_action
    
    
