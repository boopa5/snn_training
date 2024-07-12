import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os

from meta_optimizer.nn_optimizer import NNOptimizer

# ZO optimizer (UpdateRNN only)
class ZOOptimizer(NNOptimizer):

    def __init__(self, model, args=None, num_layers=1, input_dim=1, hidden_size=10, device='cuda'):
        super(ZOOptimizer, self).__init__(model, args)

        self.update_rnn = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, bias=False)
        self.outputer = nn.Linear(hidden_size, 1, bias=False)
        self.outputer.weight.data.mul_(0.1)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dim = input_dim

        self.device = device
        

    def reset_state(self, keep_states=False, model=None):
        device = self.device
        
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)
        if keep_states:
            self.h0 = Variable(self.h0.data).to(device)
            self.c0 = Variable(self.c0.data).to(device)
        else:
            def initialize_rnn_hidden_state(dim_sum, n_layers, n_params):
                h0 = Variable(torch.zeros(n_layers, n_params, dim_sum), requires_grad=True).to(device)
                return h0

            self.h0 = initialize_rnn_hidden_state(self.hidden_size, self.num_layers,
                                                  self.meta_model.get_flat_params().size(0))
            self.c0 = initialize_rnn_hidden_state(self.hidden_size, self.num_layers,
                                                  self.meta_model.get_flat_params().size(0))
            self.step = 0

    def forward(self, x):
        output1, (hn1, cn1) = self.update_rnn(x, (self.h0, self.c0))
        self.h0 = hn1
        self.c0 = cn1
        o1 = self.outputer(output1)

        return o1.squeeze()


    def meta_update(self, model, model_inputs, target, loss_fn):
            # compute the zeroth-order gradient estimate of the model
            f_x = model(data)
            loss = loss_fn(f_x, target)

            self.step += 1

            flat_grads = torch.zeros_like(model.get_params())
            for _ in range(self.q):
                u = torch.randn_like(model.get_params())  # sampled query direction
                flat_grads += self.GradientEstimate(model, data, target, u) * u
            flat_grads /= self.q

            flat_params = self.meta_model.get_flat_params()
            inputs = Variable(flat_grads.view(-1, 1).unsqueeze(1))

            # Meta update itself
            delta = self(inputs)
            flat_params = flat_params + delta

            self.meta_model.set_flat_params(flat_params)

            # Finally, copy values from the meta model to the normal one.
            self.meta_model.copy_params_to(model)
            # return self.meta_model.model, loss, f_x
            return self.meta_model.model