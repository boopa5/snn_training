import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os

from meta_optimizer.nn_optimizer import NNOptimizer
from meta_optimizer.optimizee import Optimizee, MetaModel

# ZO optimizer (UpdateRNN only)
class FOOptimizer(NNOptimizer):

    def __init__(self, model, args=None, num_layers=1, input_dim=1, hidden_size=10, partition_func=lambda _:0, device='cuda'):
        super(FOOptimizer, self).__init__(model, args)

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

        self.device = device

        
    def _init_param_masks(self, model):
        masks = [[] for _ in range(self.num_lstms)]
        for name, p in model.named_parameters():
            idx = self.param_LSTM_map[name]
            for i in range(self.num_lstms):
                if i == idx:
                    masks[idx].append(torch.ones_like(p).view(-1))
                else:
                    masks[i].append(torch.zeros_like(p).view(-1))

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

                h0s = [Variable(torch.zeros(n_layers, n_params, dim_sum), requires_grad=True).to(device) for _ in range(self.num_lstms)]
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


    def meta_update(self, optimizee: Optimizee):        
        '''Update model using gradient from model '''
        flat_params = self.meta_model.get_flat_params()

        # flat_grads = [[] for _ in range(self.num_lstms)]
        # for name, p in model.named_parameters():
        #     idx = self.param_LSTM_map[name]
        #     for i in range(self.num_lstms):
        #         flat_grads[i].append(torch.zeros_like(p).view(-1))

        #     if p.grad is not None:
        #         flat_grads[idx][-1] += p.grad.view(-1)

        
        # inputs = [Variable(flat_grad.view(-1, 1).unsqueeze(1)) for flat_grad in flat_grads]
        # deltas = [self(x, idx) for idx, x in enumerate(inputs)]

        flat_grads = []
        for name, p in optimizee.model.named_parameters():
            if p.grad is not None:
                flat_grads.append(p.grad.view(-1))
            else: 
                flat_grads.append(torch.zeros_like(p).view(-1))

        # for mask in self.param_masks:
        #     print(mask)
        flat_grads = [torch.cat(flat_grads) * mask for mask in self.param_masks]

        inputs = [Variable(flat_grad.view(-1, 1).unsqueeze(1)) for flat_grad in flat_grads]

        deltas = [self(x, idx) for idx, x in enumerate(inputs)]

        flat_params = flat_params + torch.sum(torch.stack(deltas), 0)

        self.meta_model.set_flat_params(flat_params)
        self.meta_model.copy_params_to(optimizee.model)

        return self.meta_model.model


# class FOOptimizer(NNOptimizer):

#     def __init__(self, model, args=None, num_layers=1, input_dim=1, hidden_size=10, partition_func=lambda _:0, device='cuda'):
#         super(FOOptimizer, self).__init__(model, args)

#         # self.param_LSTM_map = {name: partition_func(name) for name, _ in self.meta_model.model.named_parameters()}
#         # self.num_lstms = len(set(self.param_LSTM_map.values()))
#         # self.update_rnns = nn.ModuleList([nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, bias=False) for _ in range(self.num_lstms)])

#         self.update_rnn = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, bias=False)
#         self.outputer = nn.Linear(hidden_size, 1, bias=False)
#         self.outputer.weight.data.mul_(0.1)

#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.input_dim = input_dim

#         self.device = device
        

#     def reset_state(self, keep_states=False, model=None):
#         device = self.device
        
#         self.meta_model.reset()
#         self.meta_model.copy_params_from(model)
#         if keep_states:
#             self.h0 = Variable(self.h0.data).to(device)
#             self.c0 = Variable(self.c0.data).to(device)
#             # self.h0s = [Variable(h0.data).to(device) for h0 in self.h0s]
#             # self.c0s = [Variable(c0.data).to(device) for c0 in self.c0s]
#         else:
#             def initialize_rnn_hidden_state(dim_sum, n_layers, n_params):

#                 h0 = Variable(torch.zeros(n_layers, n_params, dim_sum), requires_grad=True).to(device)
#                 return h0

#             self.h0 = initialize_rnn_hidden_state(self.hidden_size, self.num_layers,
#                                                   self.meta_model.get_flat_params().size(0))
#             self.c0 = initialize_rnn_hidden_state(self.hidden_size, self.num_layers,
#                                                   self.meta_model.get_flat_params().size(0))
#             self.step = 0

#     def forward(self, x):
#         output1, (hn1, cn1) = self.update_rnn(x, (self.h0, self.c0))
#         self.h0 = hn1
#         self.c0 = cn1
#         o1 = self.outputer(output1)
#         # maybe multiple outputters
#         return o1.squeeze()


#     def meta_update(self, model):        
#         '''Update model using gradient from model '''
#         flat_params = self.meta_model.get_flat_params()

#         flat_grads = []
#         for name, p in model.named_parameters():
#             if p.grad is not None:
#                 flat_grads.append(p.grad.view(-1))
#             else:
#                 flat_grads.append(torch.zeros_like(p).view(-1))

#         flat_grads = torch.cat(flat_grads)
        
#         inputs = Variable(flat_grads.view(-1, 1).unsqueeze(1))

#         deltas = self(inputs)
#         # print(torch.sum(torch.stack(deltas), 0))
#         flat_params = flat_params + deltas

#         self.meta_model.set_flat_params(flat_params)
#         self.meta_model.copy_params_to(model)

#         return self.meta_model.model