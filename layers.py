import torch


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
            raise ValueError(f'Expected Tensor of dimension 4, '
                'got Tensor of dimension {dim}')
        if not self.axis < dim:
            raise IndexError(f'Axis {self.axis} outside the range for Tensor '
                'with dimension {dim}'.format(self.axis, dim))

        if self.factor == 2:
            x = x[self.slices1] + x[self.slices2]
        else:
            for _ in range(self.factor-1):
                x = x[self.slices1] + x[self.slices2]

        return x


class KMaxPool(torch.nn.Module):

    def __init__(self, k=4):
        super(KMaxPool, self).__init__()

        self.k = k

    def forward(self, x):
        dim = x.dim()
        topk, k_ind = x.topk(self.k, dim=self.axis, sorted=False)
        k_ind = k_ind.sort()[1]
        r_ind = torch.arange(dim[1])
        topk = topk[ind.sort()[1]]

        # top[torch.arange(10).int().tolist(),ind.sort()[1].transpose(0,1).tolist()].transpose(0,1)
        return x


class DynamicKMaxPool(torch.nn.Module):

    def __init__(self, layer, total_layers):
        super(DynamicKMaxPool, self).__init__()

    def forward(self, x):
        return x