import torch


class Fold(torch.nn.Module):
    def __init__(self, num=2, axis=0):
        self.num = num
        self.axis = axis
        super(Fold, self).__init__()

    def forward(self, input):
        return input
