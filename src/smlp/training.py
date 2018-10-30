from tqdm import tqdm
import torch
import pyjet.backend as J

from smlp.data import TrainingTensorDataset, SwitchboardTensorDataset
from smlp.utils import check_shape


class Trainer(object):

    def __init__(self, X, Y, network, lr=1e-1, weight_decay=1e-5,
                 batch_size=200, test_set=None):
        self.train_set = TrainingTensorDataset(X, Y)

        # Create the test set
        self.test_during_train = test_set is not None
        if self.test_during_train:
            self.test_set = SwitchboardTensorDataset(*test_set)

        self.network = network
        if J.use_cuda:
            self.network = self.network.cuda()

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr,
                                         weight_decay=self.weight_decay)

        self.batch_size = batch_size

    @torch.no_grad()
    def train_batch(self, inds, ):
        batch_size = len(inds)
        inds, batch_x, batch_y, batch_baselines, batch_weights = \
            self.train_set.get_batch(inds)

        # Run the forward
        output = self.network(batch_x).squeeze(1).long()
        check_shape(output, (batch_size,), name="Output")

        # Compute the reqard & baselines
        R = 2 * (batch_y == output).float() - 1.
        reward = batch_weights.float() * (R - batch_baselines)

        self.train_set.update_baselines(batch_baselines, R, inds)

        # Run the backward pass
        self.network.flip(reward.unsqueeze(1), batch_y)
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Update the priority weights
        # self.train_set.update_weights(batch_y, self.network.probas, inds)

        return torch.sum(R, dim=0)

    @torch.no_grad()
    def test(self, batch_size):
        assert self.test_during_train
        data_loader = self.test_set.flow(self.batch_size)
        running_acc = 0.
        for inds, in data_loader:
            # Get the test batch and evaluate
            inds, batch_x, batch_y = self.test_set.get_batch(inds)
            output = self.network(batch_x).squeeze(1)

            # Compute the accuracy
            running_acc += torch.sum((batch_y == output).float())
        return running_acc / self.test_set.num_samples

    @torch.no_grad()
    def train(self, epochs):
        for ep in range(epochs):
            print("{}/{}".format(ep + 1, epochs))

            data_loader = self.train_set.flow(self.batch_size)
            progbar = tqdm(data_loader)
            avg_reward = 0
            avg_acc = 0
            for i, (inds,) in enumerate(progbar, start=1):
                batch_reward = self.train_batch(inds).item() / len(inds)
                avg_reward += batch_reward
                avg_acc += 0.5 * (batch_reward + 1.)
                progbar.set_postfix({"r": avg_reward / i, "acc": avg_acc / i})

            # Test during train if a test set was provided
            if self.test_during_train:
                test_acc = self.test(self.batch_size)
                print("Test Accuracy:", test_acc.item())

            # Run the post ep hoook
            self.network.post_ep_hook()
        return
