import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from smlp.smlp import Switchboard


class FullyConnectedSwitchboard(Switchboard):

    def __init__(self, input_size, output_size, bias=True):
        super(FullyConnectedSwitchboard, self).__init__()

        self.weight = nn.Linear(input_size, output_size, bias=bias)
        self.weight.weight.requires_grad = False
        self.weight.bias.requires_grad = False
        self.weight.weight.zero_()
        self.weight.bias.zero_()
        # self.weight.weight.requires_grad = True
        # self.weight.bias.requires_grad = True

        self.__input_batch = None  # B x I
        self.__output_batch = None  # B x O
        self.__probability_batch = None  # B x O

        # self.__log_probability_batch = None

    @property
    def input_size(self):
        return self.weight.weight.size(1)

    @property
    def output_size(self):
        return self.weight.weight.size(0)

    @property
    def bias(self):
        return self.weight.bias is not None

    @property
    def probas(self):
        return self.__probability_batch

    # @property
    # def logprobas(self):
    #     return self.__log_probability_batch

    @property
    def inputs(self):
        return self.__input_batch

    @property
    def outputs(self):
        return self.__output_batch

    def activation(self, x):
        raise NotImplementedError()

    def log_activation(self, x):
        return NotImplementedError()

    def sample(self, probas):
        raise NotImplementedError()

    def forward(self, x):
        self.__input_batch = x
        x = self.weight(x)
        self.__probability_batch = self.activation(x)
        # self.__log_probability_batch = self.log_activation(x)

        self.__output_batch = self.sample(self.probas)
        return self.__output_batch

    def flip(self, reward, batch_y=None):
        bias_update, weight_update = self.calc_update(reward, batch_y=batch_y)
        if self.bias:
            self.weight.bias.grad = Variable(-bias_update)
        self.weight.weight.grad = Variable(-weight_update)

    def post_ep_hook(self):
        pass


class SigmoidSwitchboard(FullyConnectedSwitchboard):

    def activation(self, x):
        return torch.sigmoid(x)

    def log_activation(self, x):
        return F.logsigmoid(x), F.logsigmoid(1-x)

    def sample(self, probas):
        return torch.bernoulli(probas).detach()

    # def forward(self, x):
    #     # Tryign this
    #     self.weight.weight.data.clamp_(-1.0, 1.0)
    #     # self.weight.bias.data.clamp_(-0.5, 1.0)
    #     return super().forward(x)

    def calc_update(self, reward, batch_y=None):
        # reward is B x 1
        # outputs is B x 1
        # probas is B x m
        # inputs is B x n
        batch_size = len(reward)
        bias_grad = self.outputs - self.probas
        bias_grad *= reward

        weight_update = (bias_grad.t() @ self.inputs) / batch_size
        bias_update = torch.mean(bias_grad, dim=0)
        return bias_update, weight_update

    # def calc_update(self, reward, batch_y=None):
    #     log_prob = self.outputs * self.logprobas[0] + (1 - self.outputs) * self.logprobas[1]
    #     log_prob = torch.sum(reward * log_prob)
    #     # Multiply in the reward
    #     # log_prob = torch.sum(reward.unsqueeze(2).unsqueeze(3) * log_prob)
    #
    #     # weight_update = log_prob.backward(gradient=self.weight.weight, only_inputs=True, retain_graph=True)[0]
    #     # bias_update = torch.autograd.grad(log_prob, self.weight.bias, only_inputs=True)[0]
    #     weight_update = torch.autograd.grad(log_prob, self.weight.weight, only_inputs=True, retain_graph=True)[0]
    #     bias_update = torch.autograd.grad(log_prob, self.weight.bias, only_inputs=True)[0]
    #     return bias_update, weight_update


class SoftmaxSwitchboard(FullyConnectedSwitchboard):

    def activation(self, x):
        # print(x.shape)
        return F.softmax(x, dim=1)

    def log_activation(self, x):
        return F.log_softmax(x)

    def sample(self, probas):
        # print(probas)
        return torch.multinomial(probas, 1)

    def calc_update(self, reward, batch_y):
        # reward is B x 1
        # outputs is B x 1
        # probas is B x k
        # inputs is B x n
        batch_size = len(reward)
        bias_grad = -self.probas
        bias_grad[torch.arange(batch_size), batch_y] += 1

        weight_update = (bias_grad.t() @ self.inputs) / batch_size
        bias_update = torch.mean(bias_grad, dim=0)
        return bias_update, weight_update


class Conv2dSwitchboard(Switchboard):

    def __init__(self, input_size, output_size, kernel_size=3, bias=True):
        super(Switchboard, self).__init__()

        self.weight = nn.Conv2d(input_size, output_size, kernel_size=kernel_size, bias=bias)
        self.weight.weight.requires_grad = False
        self.weight.bias.requires_grad = False
        self.weight.weight.zero_()
        self.weight.bias.zero_()
        self.weight.weight.requires_grad = True
        self.weight.bias.requires_grad = True

        self.__input_batch = None  # B x I
        self.__output_batch = None  # B x O
        self.__probability_batch = None  # B x O

    @property
    def input_size(self):
        return self.weight.weight.size(1)

    @property
    def kernel_size(self):
        return self.weight.kernel_size

    @property
    def output_size(self):
        return self.weight.weight.size(0)

    @property
    def bias(self):
        return self.weight.bias is not None

    @property
    def probas(self):
        return self.__probability_batch

    @property
    def inputs(self):
        return self.__input_batch

    @property
    def outputs(self):
        return self.__output_batch

    def activation(self, x):
        raise NotImplementedError()

    def sample(self, probas):
        raise NotImplementedError()

    def forward(self, x):
        self.__input_batch = x
        self.__probability_batch = self.activation(self.weight(x))
        self.__output_batch = self.sample(self.probas)
        return self.__output_batch

    def flip(self, reward, batch_y=None):
        bias_update, weight_update = self.calc_update(reward, batch_y=batch_y)
        if self.bias:
            self.weight.bias.grad = Variable(-bias_update)
        self.weight.weight.grad = Variable(-weight_update)

    def post_ep_hook(self):
        pass


class SigmoidConv2dSwitchboard(Conv2dSwitchboard):

    def activation(self, x):
        return torch.sigmoid(x)

    def sample(self, probas):
        return torch.bernoulli(probas).detach()

    # def calc_update(self, reward, batch_y=None):
    #     log_prob = self.outputs * torch.log(self.probas) + (1 - self.outputs) * torch.log(1 - self.probas)
    #     log_prob = reward.unsqueeze(2).unsqueeze(3) * log_prob
    #     # Multiply in the reward
    #     # log_prob = torch.sum(reward.unsqueeze(2).unsqueeze(3) * log_prob)
    #
    #     # weight_update = log_prob.backward(gradient=self.weight.weight, only_inputs=True, retain_graph=True)[0]
    #     # bias_update = torch.autograd.grad(log_prob, self.weight.bias, only_inputs=True)[0]
    #     weight_update = torch.autograd.grad(log_prob, self.weight.weight, only_inputs=True, retain_graph=True)[0]
    #     bias_update = torch.autograd.grad(log_prob, self.weight.bias, only_inputs=True)[0]
    #     return bias_update, weight_update

    def calc_update(self, reward, batch_y=None):
        m, n = self.outputs.size(2), self.outputs.size(3)
        l, k = self.kernel_size
        B = len(reward)
        C_in = self.input_size
        C_out = self.output_size

        x_prime = torch.stack([self.inputs[:, :, i:m + i, j:n + j] for i in range(l) for j in range(k)], dim=1)  # B x kl x C_in x m x n
        x_prime = x_prime.view(B, k*l, 1, C_in, m*n)  # B x kl x 1 x C_in x mn
        v_prime = self.outputs.view(B, 1, C_out, 1, m*n)  # B x 1 x C_out x 1 x mn
        p_prime = self.probas.view(B, 1, C_out, 1, m*n)  # B x 1 x C_out x 1 x mn
        weight_update = torch.sum(x_prime * (v_prime - p_prime), dim=-1)  # B x kl x C_out x C_in
        weight_update *= reward.view(B, 1, 1, 1)

        # Average over the batch
        weight_update = torch.mean(weight_update, dim=0).view(l, k, C_out, C_in)
        weight_update = weight_update.permute(2, 3, 0, 1)  # C_out x C_in x l x k

        # Bias Update
        bias_grad = self.outputs - self.probas  # B x C_out x m x n
        bias_grad = torch.sum(bias_grad, dim=[2, 3])  # B x C_out
        bias_grad *= reward.view(B, 1)
        bias_update = torch.mean(bias_grad, dim=0)  # C _out

        return bias_update, weight_update
