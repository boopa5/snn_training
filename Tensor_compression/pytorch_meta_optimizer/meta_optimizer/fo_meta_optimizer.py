import torch
from torch.autograd import Variable

from meta_optimizer import MetaOptimizer
from utils import preprocess_gradients

class FOMetaOptimizer(MetaOptimizer):
    def __init__(self, model, num_layers, hidden_size):
        super(FOMetaOptimizer, self).__init__(model, num_layers, hidden_size)


    def meta_update(self, model_with_grads, loss):
        # First we need to create a flat version of parameters and gradients
        grads = []

        for module in model_with_grads.children():
            grads.append(module._parameters['weight'].grad.data.view(-1))
            grads.append(module._parameters['bias'].grad.data.view(-1))

        flat_params = self.meta_model.get_flat_params()
        flat_grads = preprocess_gradients(torch.cat(grads))

        inputs = Variable(torch.cat((flat_grads, flat_params.data), 1))

        # Meta update itself
        flat_params = flat_params + self(inputs)

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model