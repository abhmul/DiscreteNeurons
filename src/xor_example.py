"""
- Shallow regimes extremely quick to learn (overfitting)
- High LR required (~1)
- Using deterministic output speeds up + improves learning
- Need large batch size for quick learning
- Missing these components leads to plateua at 0.5 reward (75% acc)
"""
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pyjet.backend as J
from smlp.discrete_layer import SigmoidSwitchboard
from smlp.smlp import Switchboard

import matplotlib.pyplot as plt


class Net(Switchboard):

    def __init__(self, hidden_size=1000):
        super(Net, self).__init__()
        self.input_size = 2
        self.hidden_size = hidden_size
        self.outputs_size = 1

        self.fc1 = SigmoidSwitchboard(
            self.input_size, self.hidden_size, init=0.7)
        self.fc2 = SigmoidSwitchboard(self.hidden_size, self.outputs_size)

        self.post_init()

    @property
    def probas(self):
        return self.fc2.probas

    def forward(self, x):
        return self.fc2(self.fc1(x))

    def print_weights(self):
        print("######")
        print(self.fc1.weight.bias.data)
        print(self.fc1.weight.weight.data)
        print(self.fc2.weight.bias.data)
        print(self.fc2.weight.weight.data)


rng = np.random.RandomState(0)
X = rng.randn(1024, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(np.uint8)
xor_x = J.from_numpy(X).float()
xor_y = J.from_numpy(Y).float().unsqueeze(1)
# baseline settings


@torch.no_grad()
def train(x, y, network, optimizer, epochs=10000):
    network = network.cuda()
    baselines = J.zeros(len(x), 1) - 1.
    alpha = 0.1

    progbar = tqdm(range(epochs))
    for i in progbar:
        with torch.no_grad():
            output = network(x)
        # compute the reward & baselines
        R = 2 * (y == output).float() - 1.
        reward = R - baselines
        baselines = (1 - alpha) * baselines + alpha * R

        network.flip(reward)
        optimizer.step()
        optimizer.zero_grad()

        # Stats
        avg_reward = torch.mean(R).item()
        progbar.set_postfix({"r": avg_reward})
        # network.print_weights()
        # print("Avg Reward:", torch.mean(reward).item())
        # print()


network = Net()
# Weight decay is l2 loss
optimizer = optim.SGD(network.parameters(), lr=1e-1, weight_decay=1e-5)
train(xor_x, xor_y, network, optimizer)

# Plotting
xx, yy = np.meshgrid(np.linspace(-3, 3, 100),
                     np.linspace(-3, 3, 100))
grid_set = J.from_numpy(np.vstack((xx.ravel(), yy.ravel())).T).float()
with torch.no_grad():
    Z = network(grid_set).cpu().numpy()[:, 0]
print(np.unique(Z))
Z = Z.reshape(xx.shape)

image = plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=0,
                       linetypes='--')
plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
            edgecolors=(0, 0, 0))
plt.xticks(())
plt.yticks(())
plt.axis([-3, 3, -3, 3])
plt.colorbar(image)
plt.title("%s\n Predictions",
          fontsize=12)

plt.tight_layout()
plt.show()
