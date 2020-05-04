import numpy as np
import time

class BayesHPTuning():
    
    def __init__(self, value_priors_dict, evaluation_instance, N=2, lambdas=[5e-3,1e-3,5e-4]):
        
        self.params = {}
        for param in value_priors_dict.keys():
            self.params[param] = BayesParam(*value_priors_dict[param], prior_sample_weight=N)
            
        self.eval = evaluation_instance
        self.lambdas = lambdas
        self.history = []
    
    def step(self):
        start = time.time()
        
        HPs, HPs_probs = self.sample_params()
        print("\n"+"="*40)
        print("Configuration sampled: ")
        for p in HPs:
            print('\t'+p+' : ',HPs[p], ' - prob: %.2f'%HPs_probs[p])
        print("="*40)
        V, V_lambda = self.evaluate_params(HPs)
        self.update_params(V, V_lambda)
        
        elapsed = time.time() - start
        
        # here can be implemented conditions for logging, saving and stopping the whole thing
        self.print_step(HPs, HPs_probs, V, V_lambda, elapsed)
        
        return
    
    def sample_params(self):
        HPs = {}
        HPs_probs = {}
        for param in self.params:
            value, prob = self.params[param].sample()
            HPs[param] = value
            HPs_probs[param] = prob
        return HPs, HPs_probs
    
    def evaluate_params(self, HPs):
        # both of shape (len(lambdas), n_epochs)
        train_V_lambda, val_V_lambda = self.eval.evaluate_params(HPs, self.lambdas)
        
        d = dict(HPs=HPs, lambdas=self.lambdas, train_V_lambda=train_V_lambda, val_V_lambda=val_V_lambda)
        self.history.append(d)
        
        V = val_V_lambda.mean()
        V_lambda = val_V_lambda.mean(axis=1)
        
        return V, V_lambda
    
    def update_params(self, V, V_lambda):
        for param in self.params:
            self.params[param].update_stat(V, V_lambda)
        return
    
    def print_step(self, HPs, HPs_probs, V, V_lambda, elapsed):
        print("\n"+"="*40)
        print("Configuration sampled: ")
        for p in HPs:
            print('\t'+p+' : ',HPs[p], ' - prob: %.2f'%HPs_probs[p])
        print("="*40)
        for i, l in enumerate(self.lambdas):
            print("lambda: %.4f - V: %4f"%(self.lambdas[i],V_lambda[i]))
        print("Average V: %4f"%V)
        print("Time elapsed: %.2f s"%(elapsed))
        return
    
    def return_best_model(self):
        HPs = {}
        for param in self.params:
            probs = self.params[param].get_updated_sampling_probs()
            idx = np.argmax(probs)
            v = self.params[param].values[idx]
            HPs[param] = v
        return HPs
    
    def return_param_probs(self, param_name):
        values = self.params[param_name].values
        probs = self.params[param_name].get_updated_sampling_probs()
        return values, probs
        
            
class BayesParam():
    def __init__(self, values, priors, prior_sample_weight):
        self.values = values
        self.priors = priors
        self.prior_adv = self.compute_prior_adv()
        self.N = prior_sample_weight
        self.global_V = []
        self.last_sampled = None
        self.stat = {}
        for idx in range(len(values)):
            self.stat[idx] = {'V_lambda':[], 'V':[], 'freq':0}
            
    def sample(self):
        probs = self.get_updated_sampling_probs()
        idx = np.random.choice(np.arange(len(self.values)), p=probs)
        value = self.values[idx]
        prob = probs[idx]
        self.last_sampled = idx
        return value, prob
        
    def get_updated_sampling_probs(self):
        expected_global_V = np.mean(self.global_V)
        advantages = []
        for idx in self.stat:
            if self.stat[idx]['freq'] != 0:
                expected_Vj = np.mean(self.stat[idx]['V'])
                adv_j = (expected_Vj - expected_global_V)/expected_global_V
            else:
                adv_j = 0 # every value would do
            biased_adv_j = (self.N*self.prior_adv[idx]+self.stat[idx]['freq']*adv_j)/(self.N+self.stat[idx]['freq'])
            advantages.append(biased_adv_j)
        # sampling probs are the softmax of the biased advantages 
        advantages = np.array(advantages)
        probs = np.exp(advantages)/np.exp(advantages).sum()
        return probs
    
    def update_stat(self, V, V_lambda):
        idx = self.last_sampled
        self.stat[idx]['V'].append(V)
        self.stat[idx]['V_lambda'].append(V_lambda)
        self.stat[idx]['freq'] += 1
        self.global_V.append(V)
        return
    
    def compute_prior_adv(self):
        log_pi = np.log(self.priors)
        prior_adv = log_pi - log_pi.mean()
        return prior_adv