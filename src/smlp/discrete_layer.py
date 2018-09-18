import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from smlp.smlp import Switchboard


class FullyConnectedSwitchboard(Switchboard):

    def __init__(self, input_size, output_size, init="default"):
        super(FullyConnectedSwitchboard, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weight = nn.Linear(self.input_size, self.output_size)
        if init != "default":
            # Then init will be a number from 0-1 about average activation
            init_mean = np.log(init / (1 - init))
            nn.init.normal_(self.weight.weight.data, mean=init_mean)
            # Initialze the bias to standard normal = 2
            nn.init.normal_(self.weight.bias.data, mean=-2.)

        self.__input_batch = None  # B x I
        self.__output_batch = None  # B x O
        self.__probability_batch = None  # B x O

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
        self.weight.bias.grad = Variable(-bias_update)
        self.weight.weight.grad = Variable(-weight_update)


class SigmoidSwitchboard(FullyConnectedSwitchboard):

    def activation(self, x):
        return torch.sigmoid(x)

    def sample(self, probas):
        return torch.bernoulli(probas)

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


class SoftmaxSwitchboard(FullyConnectedSwitchboard):

    def activation(self, x):
        # print(x.shape)
        return F.softmax(x, dim=1)

    def sample(self, probas):
        # print(probas)
        return torch.multinomial(probas, 1).squeeze(1)

    def calc_update(self, reward, batch_y):
        # reward is B x 1
        # outputs is B
        # probas is B x k
        # inputs is B x n
        batch_size = len(reward)
        bias_grad = -self.probas
        bias_grad[torch.arange(batch_size), batch_y] += 1

        weight_update = (bias_grad.t() @ self.inputs) / batch_size
        bias_update = torch.mean(bias_grad, dim=0)
        return bias_update, weight_update
