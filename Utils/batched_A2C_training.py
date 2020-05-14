import torch
import torch.multiprocessing as mp
import time
import numpy as np

from Utils import test_env

def worker(worker_id, master_end, worker_end, game_params):
    master_end.close()  # Forbid worker to use the master end for messaging
    env = test_env.Sandbox(**game_params)
    np.random.seed(worker_id) # sets random seed for the environment

    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            ob_trg, reward, done, info = env.step(data)
            # Check if termination happened for time limit truncation or natural end
            if done and 'TimeLimit.truncated' in info:
                bootstrap = True
            else:
                bootstrap = False
            # ob_trg is the state used as next state for the update
            # ob is the new state used to decide the next action 
            # (different if the episode ends and another one begins)
            if done:
                ob = env.random_reset()
            else:
                ob = ob_trg
            worker_end.send((ob, reward, done, info, bootstrap, ob_trg))
        elif cmd == 'reset':
            ob = env.random_reset()
            worker_end.send(ob)
        elif cmd == 'close':
            worker_end.close()
            break
        else:
            raise NotImplementedError

class ParallelEnv:
    def __init__(self, n_train_processes, game_params):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker,
                           args=(worker_id, master_end, worker_end, game_params))
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Forbid master to use the worker end for messaging
        for worker_end in worker_ends:
            worker_end.close()

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        obs, rews, dones, infos, bootstraps, trg_obs = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos, np.stack(bootstraps), np.stack(trg_obs)

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        return np.stack([master_end.recv() for master_end in self.master_ends])

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):  # For clean up resources
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
            self.closed = True

# Aggiungere il test

def train_batched_A2C(agent, game_params, lr, n_train_processes, max_train_steps, unroll_length, test_interval=100):
    envs = ParallelEnv(n_train_processes, game_params)

    optimizer = optim.Adam(agent.parameters(), lr=lr)

    step_idx = 0
    s = envs.reset()
    while step_idx < max_train_steps:
        s_lst, a_lst, r_lst, done_lst, bootstrap_lst, s_trg_lst = list(), list(), list(), list(), list(), list()
        log_probs = []
        distributions = []
        for _ in range(unroll_length):

            a, log_prob, probs = agent.get_action(s)
            a_lst.append(a)
            log_probs.append(log_prob)
            distributions.append(probs)

            s_prime, r, done, info, bootstrap, s_trg = envs.step(a)
            s_lst.append(s)
            r_lst.append(r)
            done_lst.append(done)
            bootstrap_lst.append(bootstrap)
            s_trg_lst.append(s_trg)

            s = s_prime
            step_idx += n_train_processes

        ### Update time ###
        critic_loss, actor_loss, entropy = agent.compute_ac_loss(r_lst, log_probs, distributions, 
                                                                 s_lst, done_lst, bootstrap_lst, s_trg_lst)

        loss = (critic_loss + actor_loss).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ### 
        
        #if step_idx % PRINT_INTERVAL == 0:
        #    test(step_idx, model)

    envs.close()
    return #cose