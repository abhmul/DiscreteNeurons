from tqdm import tqdm
import torch
import torch.utils.data as data
import pyjet.backend as J

from smlp.metrics import accuracy


def create_index_dataset(size):
    return data.TensorDataset(torch.arange(size).long())


class Trainer(object):

    def __init__(self, X, Y, network, lr=1e-1, weight_decay=1e-5,
                 batch_size=1024, test_set=None):
        # TODO: Move training and test dataset to different object
        # Convert dataset to torch (keep this on cpu)
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.num_samples = len(self.X)
        assert self.num_samples == len(self.Y)
        assert self.Y.dim() == 1
        self.dataset = create_index_dataset(self.num_samples)

        # Create the test set
        self.test_during_train = test_set is not None
        if self.test_during_train:
            self.Xte, self.Yte = [torch.from_numpy(d) for d in test_set]
            self.num_test_samples = len(self.Xte)
            assert self.num_test_samples == len(self.Yte)
            assert self.Yte.dim() == 1
            self.test_dataset = create_index_dataset(self.num_test_samples)

        self.network = network
        if J.use_cuda:
            self.network = self.network.cuda()

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr,
                                         weight_decay=self.weight_decay)

        self.batch_size = batch_size
        self.__batch_inds = J.arange(self.batch_size).long()

        # Other params (keep these on the gpu)
        self.baselines = J.zeros(self.num_samples)
        self.priority_weights = J.ones(self.num_samples)
        # Use a cpu copy of the weights
        self.sampler = None
        self.set_sampling_weights()
        # constants
        self.alpha = 0.1
        self.delta = 10

    def set_sampling_weights(self):
        self.sampler = data.WeightedRandomSampler(self.priority_weights.cpu(),
                                                  self.num_samples)

    @staticmethod
    def get_batch(inds, *tensors):
        # print(inds)
        batches = [tensor[inds].cuda() if J.use_cuda else tensor[inds]
                   for tensor in tensors]
        return batches

    def get_train_batch(self, inds):
        return self.get_batch(inds, self.X, self.Y)

    def get_test_batch(self, inds):
        return self.get_batch(inds, self.Xte, self.Yte)

    @torch.no_grad()
    def train_batch(self, inds):
        batch_size = len(inds)
        batch_x, batch_y = self.get_train_batch(inds)
        batch_baselines = self.baselines[inds]

        # Run the forward
        output = self.network(batch_x)
        # print(output)
        assert tuple(output.shape) == (batch_size,), \
            "Output Shape {}".format(tuple(output.shape))

        # Compute the reqard & baselines
        R = 2 * (batch_y == output).float() - 1.
        reward = R - batch_baselines
        assert tuple(reward.shape) == (len(inds),), \
            "Reward Shape {}".format(tuple(reward.shape))
        self.baselines[inds] = (1 - self.alpha) * batch_baselines + \
            self.alpha * R

        # Run the backward pass
        self.network.flip(reward.unsqueeze(1), batch_y)
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Update the priority weights
        probs = self.network.probas
        max_p = torch.max(probs, dim=1)[0]
        min_p = torch.min(probs, dim=1)[0]
        correct_p = probs[self.__batch_inds[:batch_size], batch_y]
        self.priority_weights[inds] = ((1 - 1 / self.delta)
                                       * (correct_p - max_p) / (min_p - max_p)
                                       + 1 / self.delta) ** -1

        return torch.sum(R, dim=0)

    @torch.no_grad()
    def test(self, batch_size):
        assert self.test_during_train
        data_loader = data.DataLoader(self.test_dataset, batch_size=batch_size)
        running_acc = 0.
        for inds, in data_loader:
            # Get the test batch and evaluate
            batch_x, batch_y = self.get_test_batch(inds)
            output = self.network(batch_x)
            # print(output)
            # Compute the accuracy
            running_acc += torch.sum((batch_y == output).float())
        return running_acc / self.num_test_samples

    @torch.no_grad()
    def train(self, epochs):
        for ep in range(epochs):
            print("{}/{}".format(ep + 1, epochs))

            data_loader = data.DataLoader(self.dataset, sampler=self.sampler,
                                          batch_size=self.batch_size)
            progbar = tqdm(data_loader)
            avg_reward = 0
            avg_acc = 0
            for i, (inds,) in enumerate(progbar, start=1):
                batch_reward = self.train_batch(inds).item() / len(inds)
                avg_reward += batch_reward
                avg_acc += 0.5 * (batch_reward + 1.)
                progbar.set_postfix({"r": avg_reward / i, "acc": avg_acc / i})

                # Reset new eights for weighted sampler
                self.set_sampling_weights()

            # Test during train if a test set was provided
            if self.test_during_train:
                test_acc = self.test(self.batch_size)
                print("Test Accuracy:", test_acc.item())
        return
