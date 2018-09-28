import torch
import torch.utils.data as data

import pyjet.backend as J
from smlp.utils import check_shape


def create_index_dataset(size):
    return data.TensorDataset(torch.arange(size).long())


def get_batch(inds, *tensors):
    batches = [tensor[inds].cuda() if J.use_cuda else tensor[inds]
               for tensor in tensors]
    return batches


class SwitchboardTensorDataset(object):

    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.num_samples = len(self.X)

        assert self.num_samples == len(self.Y)
        assert self.Y.dim() == 1

        self.index_dataset = create_index_dataset(self.num_samples)

    def get_batch(self, inds):
        return get_batch(inds, self.X, self.Y)

    def flow(self, batch_size, shuffle=False):
        data_loader = data.DataLoader(self.index_dataset, shuffle=shuffle,
                                      batch_size=batch_size)
        return data_loader


class TrainingTensorDataset(SwitchboardTensorDataset):

    def __init__(self, X, Y):
        super().__init__(X, Y)

        # Other params (keep these on the gpu)
        self.baselines = J.zeros(self.num_samples)
        self.priority_weights = J.ones(self.num_samples)
        # Use a cpu copy of the weights
        self.sampler = None
        self.set_sampling_weights()
        # constants
        self.alpha = 0.1
        self.delta = 10

    def get_sampler(self):
        return data.WeightedRandomSampler(
            torch.tensor(self.priority_weights.cpu()),
            self.num_samples
        )

    def get_batch(self, inds):
        return get_batch(inds, self.X, self.Y, self.baselines)

    def flow(self, batch_size):
        data_loader = data.DataLoader(self._index_dataset,
                                      sampler=self.get_sampler(),
                                      batch_size=batch_size)
        return data_loader

    def update_baselines(self, batch_baselines, rewards, inds):
        batch_size = len(inds)
        check_shape(rewards, (batch_size,), name="Reward")
        check_shape(batch_baselines, (batch_size,), name="Baselines")

        # Rewards are B
        self.baselines[inds] = (1 - self.alpha) * batch_baselines + \
            self.alpha * rewards

    def update_weights(self, y_true, probas, inds):
        batch_size = len(inds)
        check_shape(y_true, (batch_size,), "y_true")
        check_shape(probas, (batch_size, None), "Probas")

        max_p = torch.max(probas, dim=1)[0]
        min_p = torch.min(probas, dim=1)[0]
        correct_p = probas.gather(1, y_true.unsqueeze(-1))
        self.priority_weights[inds] = ((1 - 1 / self.delta)
                                       * (correct_p - max_p) / (min_p - max_p)
                                       + 1 / self.delta) ** -1
