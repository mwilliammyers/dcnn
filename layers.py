import torch
import math
import numpy as np
import scipy.ndimage.filters

use_cuda = torch.cuda.is_available()


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


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Applies a 1D convolution over an input signal composed of several input planes.

    Args:
        input: input tensor of shape (minibatch x in_channels x iW)
        weight: filters of shape (out_channels x in_channels x kW)
        bias: optional bias of shape (out_channels). Default: None
        dilation: the spacing between kernel elements. Can be a single number or a one-element tuple (dW,). Default: 1
        groups: split input into groups, in_channels should be divisible by the number of groups. Default: 1
    """
    try:
        padding = padding[0]
        dilation = dilation[0]
        stride = stride[0]
    except TypeError:
        pass

    input = np.pad(input, [(0, 0), (0, 0), (padding, padding)], mode='constant')
    minibatch, in_channels, input_width = input.shape
    out_channels, in_channels_over_groups, weight_width = weight.shape
    out_channels_over_groups = out_channels // groups

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
        for c in range(out_channels):
            group_index = c // out_channels_over_groups * in_channels_over_groups
            for w in range(out_width):
                w_stride = w * stride
                sub = input[b, group_index:group_index + in_channels_over_groups, w_stride:w_stride + weight_width]
                out[b, c, w] = np.sum(sub * weight[c]) + bias[c]
    return out


class Conv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.save_for_backward(input, weight, bias)
        output = conv1d(input.numpy(), weight.numpy(), bias.numpy() if bias is not None else None, stride, padding, dilation, groups)
        # if bias is not None:
        #     output += bias.unsqueeze(0).expand_as(output)
        return input.new(output)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        grad_output = grad_output.data

        if ctx.needs_input_grad[0]:
            print('WEIGHT', weight.numpy())
            grad_input = scipy.signal.convolve1d(grad_output.numpy(), weight.numpy(), mode='full')
        if ctx.needs_input_grad[1]:
            grad_weight = scipy.signal.convolve1d(input.numpy(), grad_output.numpy(), mode='valid')
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return (torch.autograd.Variable(grad_output.new(grad_input)),
                torch.autograd.Variable(grad_output.new(grad_weight)),
                torch.autograd.Variable(grad_output.new(grad_bias)))


class Conv1d(torch.nn.Conv1d):
    def forward(self, input):
        return Conv1dFunction.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)


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
    np.set_printoptions(precision=4, suppress=True, floatmode='fixed')

    stride = 3
    in_channels = 6
    out_channels = 12
    padding = 4
    groups = 3

    input = torch.autograd.Variable(torch.randn((2, in_channels, 3)))
    filters = torch.autograd.Variable(torch.randn((out_channels, in_channels//groups, 3)))
    bias = torch.autograd.Variable(torch.zeros(out_channels))

    result1 = torch.nn.functional.conv1d(input, filters, stride=stride, bias=bias, padding=padding, groups=groups)
    print('INPUT', input, 'FILTERS', filters, 'TORCH RESULT', result1, sep='\n')

    result2 = conv1d(
        input.clone().data.numpy(),
        filters.clone().data.numpy(),
        stride=stride,
        bias=bias.clone().data.numpy() if bias is not None else None,
        padding=padding,
        groups=groups)
    print('RESULT', result2, result2.shape, sep='\n')

    print('MATCH?', np.allclose(result1.data.numpy(), result2))

