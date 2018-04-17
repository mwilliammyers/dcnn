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

        self.non_linearity = torch.tanh

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
        x = self.non_linearity(x)
        x = self.fc2(x)
        return x

    def params(self):
        return self.parameters()


class DCNN(torch.nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 num_classes,
                 kernel_sizes,
                 num_filters,
                 k_top=4,
                 non_linearity=torch.tanh,
                 conv1d=torch.nn.Conv1d):
        super(DCNN, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters

        self.rows = [embedding_dim // (2**x) for x in range(len(self.num_filters) + 1)]
        self.num_layers = len(self.kernel_sizes)

        self.k_top = k_top

        self.non_linearity = non_linearity
        self.fold = layers.Fold(2, 2)

        # the model
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)

        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for i, (kernel_size, num_filter, row) in enumerate(zip(self.kernel_sizes, self.num_filters, self.rows)):
            self.convs.append(
                conv1d(
                    in_channels=(self.num_filters[i - 1] if i > 0 else 1) * row,
                    out_channels=num_filter * row,
                    kernel_size=kernel_size,
                    padding=kernel_size - 1,
                    groups=row))
            self.pools.append(
                layers.DynamicKMaxPool(i + 1, self.num_layers, self.k_top)
                if i < len(self.rows) - 1 else layers.KMaxPool(self.k_top))

        self.dropout = torch.nn.Dropout()

        self.fc = torch.nn.Linear(
            in_features=self.rows[-1] * self.num_filters[-1] * self.k_top, out_features=self.num_classes)

    def forward(self, x):
        # get the sentence embedding
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        for conv, pool, num_filter in zip(self.convs, self.pools, self.num_filters):
            x = conv(x)
            x = self._to_channel_view(x, num_filter)
            x = self.fold(x)
            x = self._from_channel_view(x)
            x = pool(x)
            x = self.non_linearity(x)

        x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        # returns un-normalized log-probabilities over the classes
        return x

    def params(self):
        weight_decays = {'embedding.weight': 5e-5, 'embedding.bias': 5e-5, 'fc.weight': 5e-5, 'fc.bias': 5e-5}
        for i in range(len(self.convs)):
            val = 1.5 * 10**-(i + 5)
            weight_decays[f'convs.{i}.weight'] = val
            weight_decays[f'convs.{i}.bias'] = val
        return [{'params': v, 'weight_decay': weight_decays[k]} for k, v in self.named_parameters()]

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
