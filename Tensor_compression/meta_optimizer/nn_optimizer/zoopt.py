import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
from functools import reduce
from operator import mul

from meta_optimizer.nn_optimizer import NNOptimizer
from meta_optimizer.optimizee import Optimizee

# ZO optimizer (UpdateRNN only)
class ZOOptimizer(NNOptimizer):

    def __init__(self, model, args=None, num_layers=1, input_dim=1, hidden_size=10, partition_func=lambda _:0, q=5, mu=0.000001, device='cuda'):
        super(ZOOptimizer, self).__init__(model, args)

        self.param_LSTM_map = {name: partition_func(name) for name, _ in self.meta_model.model.named_parameters()}
        self.num_lstms = len(set(self.param_LSTM_map.values()))
        self.update_rnns = nn.ModuleList([nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, bias=False) for _ in range(self.num_lstms)])

        self.param_masks = self._init_param_masks(self.meta_model.model)

        # self.update_rnn = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, bias=False)
        self.outputer = nn.Linear(hidden_size, 1, bias=False)
        self.outputer.weight.data.mul_(0.1)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dim = input_dim

        self.q = q
        self.mu = mu

        self.device = device


    def _init_param_masks(self, model):
        masks = [[] for _ in range(self.num_lstms)]
        for name, p in model.named_parameters():
            idx = self.param_LSTM_map[name]
            for i in range(self.num_lstms):
                masks[i].append(torch.zeros_like(p).view(-1))
            masks[idx][-1] = torch.ones_like(p).view(-1)

        masks = [torch.cat(mask) for mask in masks]
        return masks

    def reset_state(self, keep_states=False, model=None):
        device = self.device

        self.meta_model.reset()
        self.meta_model.copy_params_from(model)

        if keep_states:
            # self.h0 = Variable(self.h0.data).to(device)
            # self.c0 = Variable(self.c0.data).to(device)
            self.h0s = [Variable(h0.data).to(device) for h0 in self.h0s]
            self.c0s = [Variable(c0.data).to(device) for c0 in self.c0s]

        else:
            def initialize_rnn_hidden_state(dim_sum, n_layers, n_params):

                h0s = [torch.zeros(n_layers, n_params, dim_sum, requires_grad=True).to(device) for _ in range(self.num_lstms)]
                return h0s

            self.h0s = initialize_rnn_hidden_state(self.hidden_size, self.num_layers,
                                                  self.meta_model.get_flat_params().size(0))
            self.c0s = initialize_rnn_hidden_state(self.hidden_size, self.num_layers,
                                                  self.meta_model.get_flat_params().size(0))
            self.step = 0

    def forward(self, x, idx):
        output1, (hn1, cn1) = self.update_rnns[idx](x, (self.h0s[idx], self.c0s[idx]))
        self.h0s[idx] = hn1
        self.c0s[idx] = cn1
        o1 = self.outputer(output1)
        # maybe multiple outputters
        return o1.squeeze()


    def meta_update(self, optimizee: Optimizee, model_input, loss_fn):
        '''Update model using gradient from model '''
        # with torch.no_grad():
        flat_params = optimizee.get_flat_params()

        self.step += 1

        flat_grads = torch.zeros_like(flat_params)
        
        f_x1 = optimizee.model(**model_input)
        loss_1 = loss_fn(*f_x1)

        for _ in range(self.q):
            u = torch.randn_like(self.meta_model.get_flat_params())  # sampled query direction
            flat_grads += self.GradientEstimate(optimizee, model_input, loss_1, loss_fn, u, self.mu) * u
        flat_grads /= self.q

        flat_grads = [flat_grads * mask for mask in self.param_masks]

        inputs = [Variable(flat_grad.view(-1, 1).unsqueeze(1)) for flat_grad in flat_grads]

        deltas = torch.sum(torch.stack([self(x, idx) for idx, x in enumerate(inputs)]), 0)
        # print(torch.norm(deltas))
        flat_params = flat_params + deltas

        # print(deltas.grad)

        self.meta_model.set_flat_params(flat_params)

        self.meta_model.copy_params_to(optimizee.model)

        return self.meta_model, loss_1
    

    def compute_ZO_grads(self, meta_model, model_input, loss_fn):
        
        f_x1 = meta_model.model(**model_input)
        loss1 = loss_fn(*f_x1)

        flat_grads = torch.zeros_like(meta_model.get_flat_params())
        for _ in range(self.q):
            u = torch.randn_like(meta_model.get_flat_params())  # sampled query direction
            flat_grads += self.GradientEstimate(meta_model, model_input, loss1, loss_fn, u, self.mu) * u
        flat_grads /= self.q

        return flat_grads
        # offset = 0
        # for module in meta_model.model.modules():
        #     if len(module._parameters) != 0:
        #         for key in module._parameters.keys():
        #             param_shape = module._parameters[key].size()
        #             param_flat_size = reduce(mul, param_shape, 1)
        #             module._parameters[key].grad = flat_grads[
        #                                        offset:offset + param_flat_size].view(*param_shape)
        #             offset += param_flat_size

