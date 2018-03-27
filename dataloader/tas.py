import torch
from torchtext import data as ttd
import dataloader.process_tweets as pt
import nltk


def load(batch_size=4, device=0, embedding_length=60):
    """
    Load the twitter airline sentiment dataset.

    Arguments:
        batch_size: Batch_size
        device: Device to create batches on. Use - 1 for CPU and None for
            the currently active GPU device.
    """
    tokenizer = nltk.tokenize.TweetTokenizer()
    text = ttd.Field(
        tokenize=tokenizer.tokenize,
        preprocessing=pt.preprocess(),
        lower=True,
        batch_first=True)
    label = ttd.Field(sequential=False, batch_first=True)

    fields = {'airline_sentiment': ('label', label), 'text': ('text', text)}
    dset = ttd.TabularDataset('data/twitter_airlines.csv', 'csv', fields)

    text.build_vocab(dset)
    label.build_vocab(dset)

    embed_vector = torch.rand(len(text.vocab), embedding_length)
    text.vocab.set_vectors(text.vocab.stoi, embed_vector, embedding_length)

    train, test, val = dset.split([.6, .2, .2])

    return ttd.Iterator.splits(
        (train, val, test), batch_size=batch_size, device=device)