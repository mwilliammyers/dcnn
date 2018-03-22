from torchtext import data
from torchtext import datasets
import torchtext.vocab


def load():
    text = data.Field()
    label = data.Field(sequential=False)

    train, val, test = datasets.SST.splits(
        text,
        label,
        fine_grained=True,
        train_subtrees=True,
        filter_pred=lambda ex: ex.label != 'neutral')

    f = torchtext.vocab.FastText()
    text.build_vocab(train, vectors=f)
    text.vocab.extend(f)
    label.build_vocab(train)

    return datasets.SST.iters(batch_size=4)
