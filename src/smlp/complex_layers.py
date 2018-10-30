import torch
import torch.nn as nn

from smlp.discrete_layer import SigmoidSwitchboard, SoftmaxSwitchboard
from smlp.smlp import Switchboard


class ExpansionNet(Switchboard):

    def __init__(self,
                 input_size,
                 output_size,
                 start_size=2,
                 r=2,
                 max_size=1024,
                 max_depth=3,
                 expand_per_ep=1,
                 depth_per_ep=0):
        super(ExpansionNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.start_size = start_size
        self.r = r

        self.max_size = max_size
        self.max_depth = max_depth

        # increse layer size by r every this many eps
        self.expand_per_ep = expand_per_ep
        # Increse depth by 1 every
        self.depth_per_ep = depth_per_ep
        if depth_per_ep != 0:
            raise NotImplementedError()

        self.fc1 = SigmoidSwitchboard(
            self.input_size, self.start_size)

        output_layer = SigmoidSwitchboard if self.output_size == 1 \
            else SoftmaxSwitchboard
        self.fc2 = output_layer(self.start_size, self.output_size)
        self.layers = [self.fc1, self.fc2]
        self.post_init()

    @property
    def probas(self):
        return self.layers[-1].probas

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def expand(self):
        # expand input weights
        self.fc1.weight.weight = nn.Parameter(
            # self.fc1.weight.weight.repeat(self.r, 1),
            torch.cat(
                [self.fc1.weight.weight]
                + [torch.zeros_like(self.fc1.weight.weight)
                   for _ in range(self.r-1)]
            ),
            requires_grad=self.fc1.weight.weight.requires_grad
        )
        if self.fc1.bias:
            self.fc1.weight.bias = nn.Parameter(
                # self.fc1.weight.bias.repeat(self.r),
                torch.cat(
                    [self.fc1.weight.bias]
                    + [torch.zeros_like(self.fc1.weight.bias)
                       for _ in range(self.r-1)]
                ),
                requires_grad=self.fc1.weight.bias.requires_grad
            )
        # expand output weights
        self.fc2.weight.weight = nn.Parameter(
            torch.cat(
                [self.fc2.weight.weight]
                + [torch.zeros_like(self.fc2.weight.weight)
                   for _ in range(self.r-1)],
                dim=1
            ),
            requires_grad=self.fc2.weight.weight.requires_grad
        )

    def post_ep_hook(self):
        if self.fc1.output_size * self.r < self.max_size:
            self.expand()
            print("Expanded to size:", self.fc1.output_size)
