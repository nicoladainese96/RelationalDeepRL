# RelationalDeepRL

## Summary of the material in the repo

The whole repo can be divided in:
1. Notebooks
2. RelationalModule
3. AC_modules
4. Utils

(plus other stuff that is marginal)

### Notebooks

Alphabetical order:
- A2C_A3C_sandbbox_training.ipynb: comparisons between single-process A2C, A3C and batched A2C for shared and independent architecture
- Architecture_testing.ipynb: debug of relational (BoxWorldNet) and control (OheNet) architectures layer by layer
- batched_A2C_dev.ipynb: development of training cycle and actual training of the batched A2C
- Bayesian_architecture_optimization.ipynb : class for bayesian search in HP space
- BoxWorldTesting.ipynb: code to train single-process A2C on boxworld environment
- Gated_Transformer.ipynb: development of GRU gated attention/relational block
- GatedNet_testing.ipynb: bayesian search of best HPs for GatedNet (makes use of gated attention block)
- LoadAndPlotResults.ipynb: result notebook for boxworld 
- MultiConvNet_testing.ipynb: bayesian search of best HPs for MultiplicativeConvNet (makes use of multiplicative layers and convolutional ones)
- MultiplicativeLayer.ipynb: development of various versions of the MultiplicativeLayer
- Multiprocessing.ipynb: training of A3C and single-process A2C from AC_modules 
- Results.ipynb: comprehensive report of all the results obtained with the single-process A2C
- Sandbox.ipynb: development of different state representations for Sandbox environment
- Sandbox_documentation.ipynb: check on cardinal directions mapping and print(env) functionality
- SandboxTesting.ipynb: training of all single-process A2C versions from RelationalModule
- Supervised_test.ipynb: supervised framework of the Sandbox problem

### RelationalModule

- AC_networks.py: Actor-Critic networks only
- ControlNetworks.py: networks used for the control architectures (e.g. OheNet)
- RelationalNetworks.py: networks used for relational, gated and multiplicative architectures
- MLP_AC_networks.py: Actor-Critic multi-layer perceptron networks

These files are all similar implementations of the single-process A2C for different architectures (I fixed the problem of using a different file for each architecture in the AC_modules):
- ActorCritic.py
- ControlActorCritic.py
- CoordActorCritic.py
- GatedActorCritic.py
- MultiplicativeActorCritic.py
- OheActorCritic.py

- RAdam.py: different optimizer that I tried out (unsatisfied)

### AC_modules

Here finally there is a hierarchical order in most files:
- Layers.py: all custom layers defined by me
- Networks.py: all custom networks defined by me
- ActorCriticArchitecture.py: independent and shared architecture classes for actor and critic

And then there are the 3 different Actor-Critic agents:
- AdvantageActorCritic.py: single-process A2C
- AdvantageAC_no_trg.py: A3C agent (all the multiprocessing/training part is in another script)
- BatchedA2C.py: batched A2C agent (all the multiprocessing/training part is in another script)

Main difference between this AC agents and the ones in the RelationalModule is that I took out of the class the optimization step and I just coded the functions that compute the losses from the trajectories. This is because for example in the A3C the optimization is not that straightforward, but you call loss.backward() on a CPU process and then copy the gradients of the parameters on a global model and do there the optimization. So then I adjusted the style of coding for all the architectures. This also means that the learning rate is no more automatically saved with the HPs of the model (I should fix that).

Finally I devised a Constructor (Constructor.py) kind of classes that can connect the ActorCritic with its architecture and return one or more instances of that class.

### Utils

Contains trainig, plotting, rendering and also environment scripts

- A2C_template.py: starting point for batched A2C
- A3C_training.py: A3C training for Sandbox
- batched_A2C_training.py: batched A2C training for Sandbox (except training loop that is in batched_A2C_dev.ipynb)
- HP_tuning.py: script for bayesian search 
- plot.py: plotting functions
- single_A2C_training.py: single-process A2C training for Sandbox from AC_modules
- supervised.py: all functions used to train in a supervised fashion networks on Sandbox
- test_env.py: contains Sandbox environment class
- train_agent.py: single-process A2C training for BoxWorld from RelationalModule
- train_agent_sandbox.py: single-process A2C training for Sandbox from RelationalModule
- utils.py : save and load sessions; render episodes starting from trained agents
