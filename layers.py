import torch
import math

use_cuda = torch.cuda.is_available()

def k_max_pool(x, k, axis=-1):
    top, ind = x.topk(k, dim=axis, sorted=False)
    b, d, s = top.size()
    # import pdb; pdb.set_trace()
    dim_map = torch.autograd.Variable(torch.arange(b*d), requires_grad=False)
    if use_cuda:
        dim_map = dim_map.cuda()
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

        slices = [slice(None) for _ in range(4)]
        slices[self.axis] = slice(None, None, 2)
        self.slices1 = tuple(slices)
        slices[self.axis] = slice(1, None, 2)
        self.slices2 = tuple(slices)

    def forward(self, x):

        dim = x.dim()
        if dim != 4:
            raise ValueError(f'Expected Tensor of dimension 4, got Tensor of dimension {dim}')
        if not self.axis < dim:
            raise IndexError(f'Axis {self.axis} outside the range for Tensor with dimension {dim}')

        if self.factor == 2:
            x = x[self.slices1] + x[self.slices2]
        else:
            for _ in range(self.factor - 1):
                x = x[self.slices1] + x[self.slices2]

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
