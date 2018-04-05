import numpy as np
import dataloader
import models
import layers
import logger
import torch
import tqdm
import os


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
    parser.add_argument(
        '--log',
        dest='log',
        metavar='LOG-FILE',
        type=str,
        default='logs/stats',
        help='Path to output log file')

    args = parser.parse_args()
    return args

def calc_accuracy(outputs, targets):
    correct = (outputs.data.max(dim=1)[1] == targets.data)
    return torch.sum(correct) / targets.size()[0]


if __name__ == '__main__':
    args = get_arguments()
    num_epochs = args.epochs
    batch_size = args.batch_size
    embedding_dim = 60
    device = None if torch.cuda.is_available() else -1  # None == GPU, -1 == CPU
    load_data = dataloader.twitter(embedding_dim=embedding_dim, batch_size=batch_size, device=device)

    train_iter, val_iter, test_iter = load_data()
    val_iter.sort_key = test_iter.sort_key = lambda example: len(example.text)
    num_embeddings = len(train_iter.dataset.fields['text'].vocab)
    num_classes = len(train_iter.dataset.fields['label'].vocab)

    # model = models.Model(num_embeddings, embedding_dim, num_classes)
    max_length = max(len(x.text) for x in train_iter.data())
    model = models.MLP(num_embeddings, embedding_dim, max_length, num_classes)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adagrad(model.params(), lr=.1)

    if not os.path.isdir('logs'):
        os.mkdir('logs')
    log = logger.Logger(args.log)

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
