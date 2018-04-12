import torch
import math

use_cuda = torch.cuda.is_available()


def conv1d(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Applies a 1D convolution over an input signal composed of several input planes.

    Args:
        input: input tensor of shape (minibatch x in_channels x iW)
        weight: filters of shape (out_channels x in_channels x kW)
        bias: optional bias of shape (out_channels). Default: None
        dilation: the spacing between kernel elements. Can be a single number or a one-element tuple (dW,). Default: 1
        groups: split input into groups, in_channels should be divisible by the number of groups. Default: 1
    """
    inputs = np.pad(inputs, [(0, 0), (0, 0), (padding, padding)], mode='constant')
    minibatch, in_channels, input_width = inputs.shape
    out_channels, in_channels_over_groups, weight_width = weight.shape

    if in_channels % groups != 0:
        raise ValueError('in_channels must be divisible by groups')
    if out_channels % groups != 0:
        raise ValueError('out_channels must be divisible by groups')
    if dilation != 1:
        raise NotImplementedError('dilation must be 1')

    if bias is None:
        bias = np.zeros(out_channels)

    out_width = (input_width - weight_width) // stride + 1

    out = np.empty((minibatch, out_channels, out_width))
    for b in range(minibatch):
        for w in range(out_width):
            for c in range(in_channels_over_groups):
                group_index = c // in_channels_over_groups
                w_stride = w * stride
                sub = inputs[b, group_index:group_index + in_channels_over_groups, w_stride:w_stride + weight_width]
                out[b, c, w] = np.sum(sub * weight[c]) + bias[c]
    return out


def k_max_pool(x, k, axis=-1):
    '''Perform k-max pooling operation.
    '''
    top, ind = x.topk(k, dim=axis, sorted=False)
    b, d, s = top.size()
    dim_map = torch.autograd.Variable(torch.arange(b * d), requires_grad=False)
    if use_cuda:
        dim_map = dim_map.cuda()
    # Fanciness to get the global index into the `top` tensor for each value
    # in the relative order they appeared in the input.
    offset = dim_map.view(b, d, 1).long() * s + ind.sort()[1]
    top = top.view(-1)[offset.view(-1)].view(b, d, -1)
    return top


class Fold(torch.nn.Module):
    r'''Folds an input Tensor along an axis by a folding factor.
    Expects a 4D Tensor of shape (B, C, R, L). The output will be a 4D
    Tensor with the size of the folding axis reduced by the folding
    factor.

    The folding operation collapses the Tensor along the given axis by
    adding groups of consecutive planes together. If axis=2 and factor=2,
    the output would have shape (B, C, R/2, L).

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
    r'''Applies a 1D k-max pooling over an input signal composed of
    several input planes.

    K-max pooling keeps the top k highest values from the input signal.
    The values retain their original relative ordering.

    Args:
        k: number of values to preserve through the pooling
    '''

    def __init__(self, k=4):
        super(KMaxPool, self).__init__()

        self.k = k

    def forward(self, x):
        return k_max_pool(x, self.k)


class DynamicKMaxPool(torch.nn.Module):
    r'''Applies a 1D dynamic k-max pooling over an input signal composed
    of several input planes.

    Functions similarly to KMaxPool, except that the value of k is
    computed dynamically for each input, based on the input size, the
    layer level, and the total number of layers.

    Args:
        layer: the (1 based) index of the convolution layer preceeding
               this layer
        total_layers: the total number of convolutional layers in the
                      network
        k_top: the (fixed) k-max pooling value for the final (non-dynamic)
               KMaxPool layer at the end of the network.
    '''

    def __init__(self, layer, total_layers, k_top=4):
        super(DynamicKMaxPool, self).__init__()

        self.layer = layer
        self.total_layers = total_layers
        self.k_top = k_top

    def forward(self, x):
        # compute dynamic k value
        s = x.size()[-1]
        kp = math.ceil(s * (self.total_layers - self.layer) / self.total_layers)
        k = max(self.k_top, kp)

        return k_max_pool(x, k)


if __name__ == '__main__':
    import numpy as np

    stride = 2
    np.set_printoptions(precision=4, suppress=True, floatmode='fixed')

    in_channels = 3
    out_channels = 3
    padding = 0
    groups = 2

    inputs = torch.autograd.Variable(torch.randn((2, in_channels * groups, 3)))
    filters = torch.autograd.Variable(torch.randn((out_channels * groups, in_channels, 3)))
    bias = torch.autograd.Variable(torch.zeros(out_channels * groups))

    result1 = torch.nn.functional.conv1d(inputs, filters, stride=stride, bias=bias, padding=padding, groups=groups)
    print('INPUTS', inputs, 'FILTERS', filters, 'TORCH RESULT', result1, sep='\n')

    result2 = conv1d(
        inputs.data.numpy(),
        filters.data.numpy(),
        stride=stride,
        bias=bias.data.numpy(),
        padding=padding,
        groups=groups)
    print('RESULT', result2, result2.shape, sep='\n')
