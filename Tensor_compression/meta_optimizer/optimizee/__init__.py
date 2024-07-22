import torch
from torch import nn

from functools import reduce
from operator import mul
from torch.autograd import Variable, Function

from itertools import product


class Optimizee(nn.Module):
    def __init__(self, model):
        super(Optimizee, self).__init__()
        self.model = model
        # self.loss_fn = loss_fn


    def get_flat_params(self):
        return torch.nn.utils.parameters_to_vector(self.model.parameters())


    def set_flat_params(self, flat_params):
        offset = 0
        for module in self.model.modules():
            if len(module._parameters) != 0:
                for key in module._parameters.keys():
                    param_shape = module._parameters[key].size()
                    param_flat_size = reduce(mul, param_shape, 1)
                    module._parameters[key] = flat_params[
                                               offset:offset + param_flat_size].view(*param_shape)
                    offset += param_flat_size


    def get_params_size(self):
        return self.get_params().size(0)


class MetaModel:
    def __init__(self, model):
        self.model = model


    def reset(self):
        for module in self.model.modules():
            if len(module._parameters) != 0:
                for key in module._parameters.keys():
                    module._parameters[key] = Variable(module._parameters[key].data)


    def get_flat_params(self):
        return torch.nn.utils.parameters_to_vector(self.model.parameters())


    def set_flat_params(self, flat_params):
        offset = 0
        for module in self.model.modules():
            if len(module._parameters) != 0:
                for key in module._parameters.keys():
                    param_shape = module._parameters[key].size()
                    param_flat_size = reduce(mul, param_shape, 1)
                    module._parameters[key] = flat_params[
                                               offset:offset + param_flat_size].view(*param_shape)
                    offset += param_flat_size


    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)



# from . import mnist