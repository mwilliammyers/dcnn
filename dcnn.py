import dataloader.tas as dataloader
import torch
import tqdm


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':
    model = Model()

    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_iter, val_iter, test_iter = dataloader.load()

    epoch_progress = tqdm.trange(2)
    for epoch in epoch_progress:

        running_loss = 0.0
        progress = tqdm.tqdm(train_iter)
        for batch in train_iter:
            print(batch.text, batch.label)
            # optimizer.zero_grad()
            #
            # outputs = model(batch.text)
            # loss = criterion(outputs, batch.label)
            # loss.backward()
            # optimizer.step()
            #
            # running_loss += loss.data[0]
            # progress.set_postfix(loss=running_loss)
