import dataloader.tas as dataloader
import torch
import layers
import tqdm


class Model(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Model, self).__init__()

        self.num_filters = [6, 14]
        self.kernel_size = [7, 5]

        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self.conv1 = torch.nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=self.num_filters[0] * embedding_dim,
            kernel_size=self.kernel_size[0],
            padding=self.kernel_size[0] - 1,
            groups=embedding_dim)
        self.fold1 = layers.Fold(2, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)

        x = self._to_channel_view(x, self.num_filters[0])
        x = self.fold1(x)
        x = self._from_channel_view(x)

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


if __name__ == '__main__':
    embedding_dim = 60
    batch_size = 4

    device = None if torch.cuda.is_available() else -1
    train_iter, val_iter, test_iter = dataloader.load(embedding_dim, batch_size, device=device)
    num_embeddings = len(train_iter.dataset.fields['text'].vocab)

    model = Model(num_embeddings, embedding_dim)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in tqdm.trange(1):
        running_loss = 0.0
        progress = tqdm.tqdm(train_iter)
        for batch in train_iter:
            # optimizer.zero_grad()

            outputs = model(batch.text)
            # progress.write(str(outputs))
            # loss = criterion(outputs, batch.label)
            # loss.backward()
            # optimizer.step()
            #
            # running_loss += loss.data[0]
            # progress.set_postfix(loss=running_loss)