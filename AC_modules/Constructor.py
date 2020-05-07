from AC_modules.Networks import *
from AC_modules.AdvantageActorCritic import SharedAC, IndependentAC

debug = True

class ActorCriticConstructor():
    def __init__(self, model_name, shared, *args, **kwargs):
        try:
            model = eval(model_name)
            print("Model: ", model)
        except NameError:
            print("Name of the model not found")
            return -1

        self.model = model
        self.shared = shared
        self.args = args
        self.kwargs = kwargs
        if debug:
            print("self.model: ", self.model)
            print("self.shared: ", self.shared)
            print("self.args: ", self.args)
            print("self.kwargs: ", self.kwargs)
        
    def generate_model(self):
        if self.shared:
            return SharedAC(self.model, *self.args, **self.kwargs)
        else:
            return IndependentAC(self.model, *self.args, **self.kwargs)
        
        
