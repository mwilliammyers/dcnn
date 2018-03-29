from torchtext import data
from torchtext import datasets
import torchtext.vocab


def load(batch_size=4, repeat=False, data_root='/mnt/pccfs/not_backed_up/data'):
    """
    Load the stanford sentiment treebank dataset.

    Arguments:
        batch_size: Batch_size
        device: Device to create batches on. Use - 1 for CPU and None for
            the currently active GPU device.
    """
    text = data.Field()
    label = data.Field(sequential=False)

    sst_root = data_root + '/stanford_sentiment_treebank'
    train, val, test = datasets.SST.splits(
        text,
        label,
        root=sst_root,
        fine_grained=True,
        train_subtrees=True,
        filter_pred=lambda ex: ex.label != 'neutral')

    f = torchtext.vocab.FastText(cache=data_root + '/fasttext')
    text.build_vocab(train, vectors=f)
    text.vocab.extend(f)
    label.build_vocab(train)

    return datasets.SST.iters(batch_size=batch_size, root=sst_root, repeat=repeat)
