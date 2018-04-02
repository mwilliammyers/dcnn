import torch
import math

use_cuda = torch.cuda.is_available()

def k_max_pool(x, k, axis=-1):
    top, ind = x.topk(k, dim=axis, sorted=False)
    b, d, s = top.size()
    dim_map = torch.autograd.Variable(torch.arange(b*d), requires_grad=False)
    if use_cuda:
        dim_map = dim_map.cuda()
    # Fanciness to get the global index into the `top` tensor for each value
    # in the relative order they appeared in the input.
    offset = dim_map.view(b,d,1).long() * s + ind.sort()[1]
    top = top.view(-1)[offset.view(-1)].view(b,d,-1)
    return top


class Fold(torch.nn.Module):
    r'''Folds an input Tensor along an axis by a folding factor.
    Expects a 4D Tensor of shape (B, C, R, C). The output will be a 4D
    Tensor with the size of the folding axis reduced by the folding
    factor.

    The folding operation collapses the Tensor along the given axis by
    adding groups of consecutive planes together. If axis=2 and factor=2,
    the output would have shape (B, C, R/2, C).

    The size of the folding axis should be an integer multiple of the
    folding factor.
    '''

    def __init__(self, factor=2, axis=2):
        super(Fold, self).__init__()

        self.factor = factor
        self.axis = axis
        self._setup()

    def _setup(self):

        self.slices = []
        slices = [slice(None) for _ in range(4)]
        for i in range(self.factor):
            slices[self.axis] = slice(i, None, self.factor)
            self.slices.append(tuple(slices))


    def forward(self, x):

        dim = x.dim()
        size = x.size()
        if dim != 4:
            raise ValueError(f'Expected Tensor of dimension 4, got Tensor of dimension {dim}')
        if not self.axis < dim:
            raise IndexError(f'Axis {self.axis} outside the range for Tensor with dimension {dim}')
        if size[self.axis] % self.factor != 0:
            raise ValueError(f'Axis {self.axis} of size {size[self.axis]} is not divisible by {self.factor}')

        if self.factor == 2:
            x = x[self.slices[0]] + x[self.slices[1]]
        else:
            y = x[self.slices[0]].clone()
            for i in range(1, len(self.slices)):
                y += x[self.slices[i]]
            x = y

        return x


class KMaxPool(torch.nn.Module):

    def __init__(self, k=4):
        super(KMaxPool, self).__init__()

        self.k = k

    def forward(self, x):
        return k_max_pool(x, self.k)


class DynamicKMaxPool(torch.nn.Module):

    def __init__(self, layer, total_layers, k_top=4):
        super(DynamicKMaxPool, self).__init__()

        self.layer = layer
        self.total_layers = total_layers
        self.k_top = k_top

    def forward(self, x):
        # compute dynamic k value
        s = x.size()[-1]
        kp = math.ceil(s * (self.total_layers - self.layer) / self.layer)
        k = max(self.k_top, kp)

        return k_max_pool(x, k)
