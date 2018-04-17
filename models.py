import layers
import torch
import math


class MLP(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, max_length, num_classes):
        super(MLP, self).__init__()

        self.embedding_dim = embedding_dim
        self.max_length = max_length

        self.input_size = embedding_dim * max_length
        self.hidden_size = 2 * self.input_size
        self.output_size = num_classes

        self.nonlin = torch.tanh

        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self.fc1 = torch.nn.Linear(in_features=self.input_size, out_features=self.hidden_size)
        self.fc2 = torch.nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, x):
        b, n = x.size()
        pad = (0, (self.max_length - n) * self.embedding_dim)
        x = self.embedding(x)
        x = x.view(b, -1)
        x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
        x = self.fc1(x)
        x = self.nonlin(x)
        x = self.fc2(x)
        return x

    def params(self):
        return self.parameters()


class _DCNNBase(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_classes, num_layers=2, k_top=4):
        super(_DCNNBase, self).__init__()

        self.num_layers = num_layers
        self.k_top = k_top
        self.num_classes = num_classes

    def make_bias(self, size):
        n = size[0]
        for k in size[1:]:
            n *= k
        stdv = 1. / math.sqrt(n)
        bias = torch.nn.Parameter(torch.Tensor(*size))
        bias.data.uniform_(-stdv, stdv)
        return bias

    def _to_channel_view(self, x, channels):
        # Get a 4D (batch, channels, embed, length) view of the 3D
        # (batch, embed*channels, length) Tensor. Note that axis 1 of the input
        # has the form [row1-filter1, row1-filter2, ..., rowd-filterk] where d
        # is the embedding length and k is the number of conv filters. We want
        # each channel to represent all the input rows, so we need to collate
        # appropriately
        (b, d, s), k = x.size(), channels
        x = x.view(b, d // k, k, s).permute(0, 2, 1, 3)
        return x

    def _from_channel_view(self, x):
        # Get a 3D (batch, embed*channels, length) view of the 4D
        # (batch, channels, embed, length) Tensor. See self._to_channel_view
        # for more info.
        b, k, d, s = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b, d * k, s)
        return x


class DCNN(_DCNNBase):
    def __init__(self, num_embeddings, embedding_dim, num_classes, num_layers=2, k_top=4):
        super(DCNN, self).__init__(num_embeddings, embedding_dim, num_classes, num_layers, k_top)

        self.num_filters = [6, 14]
        self.kernel_size = [7, 5]
        self.rows = [embedding_dim, embedding_dim // 2, embedding_dim // 4]

        self.nonlin = torch.tanh

        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self.conv1 = torch.nn.Conv1d(
            in_channels=self.rows[0],
            out_channels=self.num_filters[0] * self.rows[0],
            kernel_size=self.kernel_size[0],
            padding=self.kernel_size[0] - 1,
            groups=self.rows[0])
        self.fold1 = layers.Fold(2, 2)
        self.dkmpool1 = layers.DynamicKMaxPool(1, self.num_layers, self.k_top)
        self.bias1 = self.make_bias((self.num_filters[0] * self.rows[1], 1))

        self.conv2 = torch.nn.Conv1d(
            in_channels=self.num_filters[0] * self.rows[1],
            out_channels=self.num_filters[1] * self.rows[1],
            kernel_size=self.kernel_size[1],
            padding=self.kernel_size[1] - 1,
            groups=self.rows[1])
        self.fold2 = layers.Fold(2, 2)
        self.kmaxpool = layers.KMaxPool(self.k_top)
        self.bias2 = self.make_bias((self.num_filters[1] * self.rows[2], 1))

        self.dropout = torch.nn.Dropout()

        self.fc = torch.nn.Linear(
            in_features=self.rows[2] * self.num_filters[1] * self.k_top, out_features=self.num_classes)

    def forward(self, x):
        # get the sentence embedding
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        # first conv-fold-pool block
        x = self.conv1(x)
        x = self._to_channel_view(x, self.num_filters[0])
        x = self.fold1(x)
        x = self._from_channel_view(x)
        x = self.dkmpool1(x)
        x = x + self.bias1
        x = self.nonlin(x)
        # second conv-fold-pool block
        x = self.conv2(x)
        x = self._to_channel_view(x, self.num_filters[1])
        x = self.fold2(x)
        x = self._from_channel_view(x)
        x = self.kmaxpool(x)
        x = x + self.bias2
        x = self.nonlin(x)

        x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        # returns un-normalized log-probabilities over the classes
        return x

    def params(self):
        weight_decays = {
            'embedding': 5e-5,
            'conv1': 1.5e-5,
            'bias1': 1.5e-5,
            'conv2': 1.5e-6,
            'bias2': 1.5e-6,
            'fc': 5e-5
        }
        return [{'params': v, 'weight_decay': weight_decays[k.split('.')[0]]} for k, v in self.named_parameters()]


class DeepDCNN(_DCNNBase):
    def __init__(self, num_embeddings, embedding_dim, num_classes, num_layers=2, k_top=4):
        super(DeepDCNN, self).__init__(num_embeddings, embedding_dim, num_classes, num_layers, k_top)

        self.num_filters = [10, 14, 18]
        self.kernel_size = [7, 5, 5]
        # self.rows = [embedding_dim, embedding_dim // 2, embedding_dim // 4]
        self.rows = [embedding_dim // (2**x) for x in range(len(self.num_filters) + 1)]

        self.nonlin = torch.tanh
        self.fold = layers.Fold(2, 2)

        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self.conv1 = torch.nn.Conv1d(
            in_channels=self.rows[0],
            out_channels=self.num_filters[0] * self.rows[0],
            kernel_size=self.kernel_size[0],
            padding=self.kernel_size[0] - 1,
            groups=self.rows[0])
        self.dkmpool1 = layers.DynamicKMaxPool(1, self.num_layers, self.k_top)

        self.conv2 = torch.nn.Conv1d(
            in_channels=self.num_filters[0] * self.rows[1],
            out_channels=self.num_filters[1] * self.rows[1],
            kernel_size=self.kernel_size[1],
            padding=self.kernel_size[1] - 1,
            groups=self.rows[1])
        self.dkmpool2 = layers.DynamicKMaxPool(2, self.num_layers, self.k_top)

        self.conv3 = torch.nn.Conv1d(
            in_channels=self.num_filters[1] * self.rows[2],
            out_channels=self.num_filters[2] * self.rows[2],
            kernel_size=self.kernel_size[2],
            padding=self.kernel_size[2] - 1,
            groups=self.rows[2])
        self.dkmpool3 = layers.DynamicKMaxPool(3, self.num_layers, self.k_top)

        self.conv4 = torch.nn.Conv1d(
            in_channels=self.num_filters[2] * self.rows[3],
            out_channels=self.num_filters[3] * self.rows[3],
            kernel_size=self.kernel_size[3],
            padding=self.kernel_size[3] - 1,
            groups=self.rows[3])
        self.kmaxpool = layers.KMaxPool(self.k_top)

        self.dropout = torch.nn.Dropout()

        self.fc = torch.nn.Linear(
            in_features=self.rows[-1] * self.num_filters[-1] * self.k_top, out_features=self.num_classes)

    def forward(self, x):
        # get the sentence embedding
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        # first conv-fold-pool block
        x = self.conv1(x)
        x = self._to_channel_view(x, self.num_filters[0])
        x = self.fold(x)
        x = self._from_channel_view(x)
        x = self.dkmpool1(x)
        x = self.nonlin(x)
        # second conv-fold-pool block
        x = self.conv2(x)
        x = self._to_channel_view(x, self.num_filters[1])
        x = self.fold(x)
        x = self._from_channel_view(x)
        x = self.dkmpool2(x)
        x = self.nonlin(x)
        # third conv-fold-pool block
        x = self.conv3(x)
        x = self._to_channel_view(x, self.num_filters[2])
        x = self.fold(x)
        x = self._from_channel_view(x)
        x = self.dkmpool3(x)
        x = self.nonlin(x)
        # fourth conv-fold-pool block
        x = self.conv4(x)
        x = self._to_channel_view(x, self.num_filters[3])
        x = self.fold(x)
        x = self._from_channel_view(x)
        x = self.kmaxpool(x)
        x = self.nonlin(x)

        x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        # returns un-normalized log-probabilities over the classes
        return x

    def params(self):
        weight_decays = {
            'embedding': 5e-5,
            'conv1': 1.5e-5,
            'conv2': 1.5e-6,
            'conv3': 1.5e-7,
            'conv4': 1.5e-8,
            'fc': 5e-5
        }  # yapf: disable
        return [{'params': v, 'weight_decay': weight_decays[k.split('.')[0]]} for k, v in self.named_parameters()]


class DCNNReLU(DCNN):
    def __init__(self, num_embeddings, embedding_dim, num_classes, num_layers=2, k_top=4):
        super(DCNNReLU, self).__init__(num_embeddings, embedding_dim, num_classes, num_layers=2, k_top=4)

        self.nonlin = torch.nn.ReLU()


class DCNNLeakyReLU(DCNN):
    def __init__(self, num_embeddings, embedding_dim, num_classes, num_layers=2, k_top=4):
        super(DCNNLeakyReLU, self).__init__(num_embeddings, embedding_dim, num_classes, num_layers=2, k_top=4)

        self.nonlin = torch.nn.LeakyReLU()


class CustomDCNN(DCNN):
    def __init__(self, num_embeddings, embedding_dim, num_classes, num_layers=2, k_top=4):
        super(CustomDCNN, self).__init__(num_embeddings, embedding_dim, num_classes, num_layers=2, k_top=4)

        self.conv1 = layers.Conv1d(
            in_channels=self.rows[0],
            out_channels=self.num_filters[0] * self.rows[0],
            kernel_size=self.kernel_size[0],
            padding=self.kernel_size[0] - 1,
            groups=self.rows[0])

        self.conv2 = layers.Conv1d(
            in_channels=self.num_filters[0] * self.rows[1],
            out_channels=self.num_filters[1] * self.rows[1],
            kernel_size=self.kernel_size[1],
            padding=self.kernel_size[1] - 1,
            groups=self.rows[1])
