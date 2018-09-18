import torch.nn as nn


# Template design pattern
class Switchboard(nn.Module):

    def __init__(self):
        super(Switchboard, self).__init__()
        self.switchboard_modules = []
        self.post_init()

    @property
    def probas(self):
        raise NotImplementedError()

    @property
    def inputs(self):
        raise NotImplementedError()

    @property
    def outputs(self):
        raise NotImplementedError()

    def post_init(self):
        """Collects all Switchboards that are part of this switchboard"""
        self.switchboard_modules = [
            module for module in self.children()
            if isinstance(module, Switchboard)
        ]

    def flip(self, reward, batch_y=None):
        if not self.switchboard_modules:
            raise AttributeError("No submodules (did you call post_init)?")
        for module in self.switchboard_modules:
            module.flip(reward, batch_y=batch_y)
