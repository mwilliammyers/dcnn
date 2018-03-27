import dataloader.tas as dataloader
import torch
import tqdm


class Model(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Model, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self.conv1 = torch.nn.Conv1d(embedding_dim, 6 * embedding_dim, kernel_size=7, padding=6, groups=embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        return x


if __name__ == '__main__':
    embedding_dim = 60
    batch_size = 4

    train_iter, val_iter, test_iter = dataloader.load(embedding_dim, batch_size)
    num_embeddings = len(train_iter.dataset.fields['text'].vocab)

    model = Model(num_embeddings, embedding_dim).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epoch_progress = tqdm.trange(1)
    for epoch in epoch_progress:
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
