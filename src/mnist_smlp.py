from __future__ import print_function

import _pickle as cPickle
import gzip
import wget
import numpy as np
import matplotlib.pyplot as plt

from smlp.smlp import Switchboard
from smlp.discrete_layer import SigmoidSwitchboard, SoftmaxSwitchboard
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

# Combine the training and validation sets
xtr = np.concatenate([xtr, xval], axis=0)
ytr = np.concatenate([ytr, yval], axis=0)

print(np.max(xtr))
print("Training Data Shape: ", xtr.shape)
print("Training Labels Shape: ", ytr.shape)

# Visualize an image
# ind = np.random.randint(xtr.shape[0])
# plt.imshow(xtr[ind, 0, :, :], cmap='gray')
# plt.title("Digit = %s" % ytr[ind])
# plt.show()


class Net(Switchboard):

    def __init__(self, hidden_size=20000):
        super(Net, self).__init__()
        self.input_size = 28 * 28
        self.hidden_size = hidden_size
        self.outputs_size = 10

        self.fc1 = SigmoidSwitchboard(
            self.input_size, self.hidden_size, init=0.7)
        self.fc2 = SoftmaxSwitchboard(self.hidden_size, self.outputs_size)

        self.post_init()

    @property
    def probas(self):
        return self.fc2.probas

    def forward(self, x):
        x = x.view(-1, self.input_size)
        return self.fc2(self.fc1(x))


network = Net()
trainer = Trainer(xtr, ytr, network, lr=1e-1, weight_decay=1e-5,
                  test_set=(xte, yte))

trainer.train(1000)
