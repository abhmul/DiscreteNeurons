import torch


def accuracy(y_true, y_pred):
    assert y_true.dim() == y_pred.dim() == 1
    return torch.mean((y_pred == y_true).float())
