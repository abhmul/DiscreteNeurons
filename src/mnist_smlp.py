from __future__ import print_function

import _pickle as cPickle
import gzip
import wget
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from smlp.smlp import Switchboard
from smlp.discrete_layer import SigmoidSwitchboard, SoftmaxSwitchboard, SigmoidConv2dSwitchboard
from smlp.complex_layers import ExpansionNet
from smlp.training import Trainer


# Load the dataset
try:
    f = gzip.open('mnist_py3k.pkl.gz', 'rb')
except:
    print("Could not find MNIST, downloading the dataset")
    wget.download(
        "http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist_py3k.pkl.gz")
    f = gzip.open('mnist_py3k.pkl.gz', 'rb')
(xtr, ytr), (xval, yval), (xte, yte) = cPickle.load(f)
# Need to convert to keras format
f.close()

xtr = xtr.reshape((-1, 1,  28, 28))  # Should be (Channel Height, Width)
xval = xval.reshape((-1, 1,  28, 28))  # Should be (Channel Height, Width)
xte = xte.reshape((-1, 1,  28, 28))  # Should be (Channel Height, Width)

# Combine the training and validation sets
xtr = np.concatenate([xtr, xval], axis=0)
ytr = np.concatenate([ytr, yval], axis=0)

# Binarize the input
xtr = (xtr > 0.5).astype(xtr.dtype)
xte = (xte > 0.5).astype(xte.dtype)

print(np.max(xtr))

print("Training Data Shape: ", xtr.shape)
print("Training Labels Shape: ", ytr.shape)

# Visualize an image
# ind = np.random.randint(xtr.shape[0])
# plt.imshow(xtr[ind, 0, :, :], cmap='gray')
# plt.title("Digit = %s" % ytr[ind])
# plt.show()


class Net(Switchboard):

    def __init__(self, ):
        super(Net, self).__init__()
        self.input_size = 28 * 28
        self.output_size = 10
        hidden_sizes = [500] # , 250]

        self.fc1 = SigmoidSwitchboard(self.input_size, hidden_sizes[0])

        # self.fc2 = SigmoidSwitchboard(hidden_sizes[0], hidden_sizes[1])
        # self.fc2.weight.weight.zero_()
        # self.fc2.weight.bias.zero_()

        # # self.drop = nn.Dropout(0.5)
        self.final = SoftmaxSwitchboard(hidden_sizes[-1], self.output_size)
        # self.final.weight.weight.zero_()
        # self.final.weight.bias.zero_()

        # self.net = ExpansionNet(
        #     self.input_size,
        #     self.output_size
        # )
        self.post_init()

        # print(self.fc1.weight.weight)
        # print(self.fc2.weight.weight)

    @property
    def probas(self):
        return self.net.probas

    def forward(self, x):
        x = x.view(-1, self.input_size)
        # x = self.net(x)
        x = self.fc1(x)
        # x = self.fc2(x)
        x = self.final(x)
        return x

    def post_ep_hook(self):
        print("=========")
        print(self.fc1.weight.weight.data.max())
        print(self.fc1.weight.weight.data.min())
        # print(self.fc1.weight.bias.data)
        # print("=========")
        # print(self.fc2.weight.weight.data)
        # print(self.fc2.weight.bias.data)

class Net2(Switchboard):

    def __init__(self, ):
        super(Net2, self).__init__()
        self.input_size = 28 * 28
        self.output_size = 10
        self.kernel_size = 9
        hidden_sizes = [500]

        self.conv1 = SigmoidConv2dSwitchboard(1, hidden_sizes[0], kernel_size=self.kernel_size)

        self.flat_size = (28 - self.kernel_size + 1) ** 2 * hidden_sizes[0]
        self.final = SoftmaxSwitchboard(self.flat_size, self.output_size)
        self.final.weight.weight.zero_()
        self.final.weight.bias.zero_()
        self.post_init()

    @property
    def probas(self):
        return self.net.probas

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, self.flat_size)
        x = self.final(x)
        return x

    # def post_ep_hook(self):
        # print("=========")
        # print(self.conv1.weight.weight.data.max())
        # print(self.fc1.weight.weight.data.min())
        # print(self.fc1.weight.bias.data)
        # print("=========")
        # print(self.fc2.weight.weight.data)
        # print(self.fc2.weight.bias.data)


network = Net2()
trainer = Trainer(xtr, ytr, network, lr=1e-1,# weight_decay=1e-5,
                  test_set=(xte, yte))

trainer.train(100)

# Plot the weights
# weights = network.fc1.weight.weight.view(-1, 28, 28).cpu().numpy()
# assert np.all(weights[0].flatten() == network.fc1.weight.weight[0].cpu().numpy())
# for weight in weights:
#     plt.imshow(weight, cmap='gray')
#     plt.show()
