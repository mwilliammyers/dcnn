import numpy as np
import dataloader
import torch
import layers
import logger
import tqdm
import os


class Model(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_classes, num_layers=2, k_top=4):
        super(Model, self).__init__()

        self.num_filters = [6, 14]
        self.kernel_size = [7, 5]
        self.rows = [embedding_dim, embedding_dim // 2, embedding_dim // 4]

        self.num_layers = num_layers
        self.k_top = k_top
        self.num_classes = num_classes

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

        self.conv2 = torch.nn.Conv1d(
            in_channels=self.num_filters[0] * self.rows[1],
            out_channels=self.num_filters[1] * self.rows[1],
            kernel_size=self.kernel_size[1],
            padding=self.kernel_size[1] - 1,
            groups=self.rows[1])
        self.fold2 = layers.Fold(2, 2)
        self.kmaxpool = layers.KMaxPool(self.k_top)

        self.dropout = torch.nn.Dropout()

        self.fc = torch.nn.Linear(
            in_features=self.rows[2] * self.num_filters[1] * self.k_top,
            out_features=self.num_classes)

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
        x = self.nonlin(x)
        # second conv-fold-pool block
        x = self.conv2(x)
        x = self._to_channel_view(x, self.num_filters[1])
        x = self.fold2(x)
        x = self._from_channel_view(x)
        x = self.kmaxpool(x)
        x = self.nonlin(x)

        x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        # returns un-normalized log-probabilities over the classes
        return x

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


def get_arguments():
    import argparse
    parser = argparse.ArgumentParser('Dynamic CNN in PyTorch')

    parser.add_argument(
        '--num-epochs',
        dest='epochs',
        metavar='EPOCHS',
        type=int,
        default=5,
        help='Number of epochs to train for.')  # yapf: disable
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        metavar='BATCH-SIZE',
        type=int,
        default=4,
        help='Size of a mini batch.')  # yapf: disable

    args = parser.parse_args()
    return args

def calc_accuracy(outputs, targets):
    correct = (outputs.data.max(dim=1)[1] == targets.data)
    return torch.sum(correct) / targets.size()[0]


if __name__ == '__main__':
    args = get_arguments()
    num_epochs = args.epochs
    batch_size = args.batch_size
    # TODO: make configurable from CLI?
    weight_decays = {'embedding': 5e-5, 'conv1': 1.5e-5, 'conv2': 1.5e-6, 'fc': 5e-5}

    embedding_dim = 60
    device = None if torch.cuda.is_available() else -1  # None == GPU, -1 == CPU
    load_data = dataloader.twitter(embedding_dim=embedding_dim, batch_size=batch_size, device=device)

    train_iter, val_iter, test_iter = load_data()
    val_iter.sort_key = test_iter.sort_key = lambda example: len(example.text)
    num_embeddings = len(train_iter.dataset.fields['text'].vocab)
    num_classes = len(train_iter.dataset.fields['label'].vocab)

    model = Model(num_embeddings, embedding_dim, num_classes)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()

    params = [{'params': v, 'weight_decay': weight_decays[k.split('.')[0]]} for k, v in model.named_parameters()]
    optimizer = torch.optim.Adagrad(params, lr=.1)

    if not os.path.isdir('logs'):
        os.mkdir('logs')
    log = logger.Logger('logs/stats')

    # stats == [train_loss, train_acc, test_loss, test_acc]
    stats = np.zeros(4, dtype='float64')
    update_period = 200
    with tqdm.tqdm(train_iter, total=len(train_iter) * num_epochs) as progress:
        stats[:2] = 0
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()

            outputs = model(batch.text)
            loss = criterion(outputs, batch.label)
            loss.backward()
            optimizer.step()

            stats[0] += loss.data[0]
            stats[1] += calc_accuracy(outputs, batch.label)

            progress.update()
            if i % update_period == update_period - 1:
                stats[2:] = 0
                for j, batch in enumerate(val_iter):
                    outputs = model(batch.text)
                    loss = criterion(outputs, batch.label)
                    stats[2] += loss.data[0]
                    stats[3] += calc_accuracy(outputs, batch.label)
                stats[:2] /= update_period
                stats[2:] /= len(val_iter)
                log.log(stats.astype('float32'))
                progress.set_postfix(val_loss=stats[2], train_loss=stats[0])
                train_loss = 0
            if train_iter.epoch >= num_epochs:
                break
