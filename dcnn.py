import numpy as np
import dataloader
import models
import time
import torch
import tqdm
from tensorboardX import SummaryWriter


def get_arguments():
    import argparse
    model_choices = ['dcnn', 'mlp']
    non_linearity_choices = ['tanh', 'relu', 'leaky-relu']
    optim_choices = ['adagrad', 'adadelta', 'adam']
    dataset_choices = ['twitter', 'twitter-large', 'yelp']

    parser = argparse.ArgumentParser('Dynamic CNN in PyTorch', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--current-run',
        dest='current_run',
        metavar='CURRENT_RUN',
        type=int,
        default=0,
        help='The current run')  # yapf: disable
    parser.add_argument(
        '--num-epochs',
        dest='num_epochs',
        metavar='EPOCHS',
        type=int,
        default=15,
        help='Number of epochs to train for')  # yapf: disable
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        metavar='BATCH-SIZE',
        type=int,
        default=64,
        help='Size of a mini batch')  # yapf: disable
    parser.add_argument(
        '--kernel-sizes',
        dest='kernel_sizes',
        metavar='KERNEL-SIZE',
        type=int,
        nargs='*',
        default=[7, 5],
        help='Size of the kernels per layer; formatted as a space delimited list')  # yapf: disable
    parser.add_argument(
        '--num-filters',
        dest='num_filters',
        metavar='NUM-FILTER',
        type=int,
        nargs='*',
        default=[6, 14],
        help='The number of filters at each layer; formatted as a space delimited list')  # yapf: disable
    parser.add_argument(
        '--log',
        dest='log',
        metavar='LOG-FILE',
        type=str,
        default='logs/stats',
        help='Path to output log file')  # yapf: disable
    parser.add_argument(
        '--model',
        dest='model',
        metavar='MODEL-TYPE',
        type=str,
        default='dcnn',
        choices=model_choices,
        help=f'Model to use. One of {model_choices}')  # yapf: disable
    parser.add_argument(
        '--lr',
        dest='learning_rate',
        metavar='LEARNING-RATE',
        type=float,
        default=0.05,
        help='Learning rate')  # yapf: disable
    parser.add_argument(
        '--embed-dim',
        dest='embedding_dim',
        metavar='EMBED-DIMENSION',
        type=int,
        default=60,
        help='Dimension of the word embeddings')  # yapf: disable
    parser.add_argument(
        '--eval-period',
        dest='eval_period',
        metavar='NUM-BATCHES',
        type=int,
        default=200,
        help='Number of training batches between validation evals')  # yapf: disable
    parser.add_argument(
        '--non-linearity',
        dest='non_linearity',
        metavar='FUNCTION',
        default=non_linearity_choices[0],
        choices=non_linearity_choices,
        help=f'Non linearity function. One of {non_linearity_choices}')  # yapf: disable
    parser.add_argument(
        '--optim',
        dest='optim',
        metavar='OPTIMIZER-ALGORITHM',
        default=optim_choices[0],
        choices=optim_choices,
        help=f'Optimization algorithm. One of {optim_choices}')  # yapf: disable
    parser.add_argument(
        '--dataset',
        dest='dataset',
        metavar='DATASET',
        default=dataset_choices[0],
        choices=dataset_choices,
        help=f'Dataset. One of {dataset_choices}')  # yapf: disable
    parser.add_argument(
        '--track-mistakes',
        dest='track_mistakes',
        action='store_true',
        default=False,
        help='Show counts of mis-predicted labels')  # yapf: disable
    return parser.parse_args()


def get_model(args, num_embeddings, num_classes):
    non_linearities = {'tanh': torch.tanh, 'relu': torch.nn.ReLU(), 'leaky-relu': torch.nn.LeakyReLU()}
    if args.model == 'dcnn':
        model = models.DCNN(
            num_embeddings,
            args.embedding_dim,
            num_classes,
            kernel_sizes=args.kernel_sizes,
            num_filters=args.num_filters,
            non_linearity=non_linearities[args.non_linearity])
    elif args.model == 'mlp':
        max_length = max(len(x.text) for x in train_iter.data())
        model = models.MLP(num_embeddings, args.embedding_dim, max_length, num_classes)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def get_optim(args, parameters):
    if args.optim == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=args.learning_rate)
    if args.optim == 'adadelta':
        return torch.optim.Adadelta(parameters)
    elif args.optim == 'adam':
        return torch.optim.Adam(parameters, lr=args.learning_rate)


def calc_accuracy(outputs, targets):
    correct = (outputs.data.max(dim=1)[1] == targets.data)
    return torch.sum(correct) / targets.size()[0]


def compute_confusion(model, val_iter):
    print('Evaluating mistakes...')
    mistakes = {k: 0 for k in val_iter.dataset.fields['label'].vocab.itos}
    confusion = np.zeros((3, 3))
    for batch in val_iter:
        outputs = model(batch.text)
        loss = criterion(outputs, batch.label)
        targets = batch.label
        predictions = outputs.data.max(dim=1)[1]
        correct = (predictions == targets.data)
        correct = correct.cpu().numpy().astype('bool')
        label = targets.data.cpu().numpy().tolist()
        predictions = predictions.cpu().numpy().tolist()
        for lab, pred in zip(label, predictions):
            confusion[lab, pred] += 1
    print('Confusion matrix:')
    print(confusion)
    return confusion


if __name__ == '__main__':
    args = get_arguments()

    device = None if torch.cuda.is_available() else -1  # None == GPU, -1 == CPU
    if args.dataset == 'twitter':
        load_data = dataloader.twitter(embedding_dim=args.embedding_dim, batch_size=args.batch_size, device=device)
    elif args.dataset == 'twitter-large':
        load_data = dataloader.twitter(
            embedding_dim=args.embedding_dim,
            path='data/twitter.training.csv',
            fields=('text', 'label'),
            batch_size=args.batch_size,
            device=device)
    elif args.dataset == 'yelp':
        load_data = dataloader.yelp(embedding_dim=args.embedding_dim, batch_size=args.batch_size, device=device)

    train_iter, val_iter, test_iter = load_data()
    val_iter.sort_key = test_iter.sort_key = lambda example: len(example.text)

    model = get_model(
        args,
        num_embeddings=len(train_iter.dataset.fields['text'].vocab),
        num_classes=len(train_iter.dataset.fields['label'].vocab))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optim(args, model.params())

    log_file = f"{args.log}_run-{args.current_run}_{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())}"
    writer = SummaryWriter(log_file)

    writer.add_text('hyperparameters', str(args.__dict__))

    # `stats` has form [train_loss, train_acc, validation_loss, validation_acc]
    stats = np.zeros(4, dtype='float64')
    desc = f'run {args.current_run}'
    with tqdm.tqdm(train_iter, total=len(train_iter) * args.num_epochs, position=args.current_run, desc=desc) as pbar:
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()

            outputs = model(batch.text)
            loss = criterion(outputs, batch.label)
            loss.backward()
            optimizer.step()

            stats[0] += loss.data[0]
            stats[1] += calc_accuracy(outputs, batch.label)

            pbar.update()
            if (i % args.eval_period) == (args.eval_period - 1):
                for batch in val_iter:
                    outputs = model(batch.text)
                    loss = criterion(outputs, batch.label)
                    stats[2] += loss.data[0]
                    stats[3] += calc_accuracy(outputs, batch.label)
                stats[:2] /= args.eval_period
                stats[2:] /= len(val_iter)

                writer.add_scalar('stats/train_loss', stats[0], i)
                writer.add_scalar('stats/train_acc', stats[1], i)
                writer.add_scalar('stats/validation_loss', stats[2], i)
                writer.add_scalar('stats/validation_acc', stats[3], i)

                for name, param in model.named_parameters():
                    writer.add_histogram(name, param, i)

                pbar.set_postfix(train_loss=stats[0], train_acc=stats[1], val_loss=stats[2], val_acc=stats[3])
                stats[:] = 0
            if train_iter.epoch >= args.num_epochs:
                break

    if args.track_mistakes:
        confusion = compute_confusion(model, val_iter)

    writer.scalar_dict['hyperparameters'] = args.__dict__
    title = f"{args.log.split('/')[-1].split('_')[0]}_{args.model}_{args.dataset}_run-{args.current_run}"
    writer.scalar_dict['title'] = title
    writer.export_scalars_to_json(f'{log_file}.json')
    writer.close()
