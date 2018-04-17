import torch
import torchtext.data
import torchtext.datasets


def load(path,
         format,
         fields,
         tokenize=str.split,
         preprocessing=None,
         embedding_dim=60,
         batch_size=4,
         device=None,
         skip_header=False):
    """Load a tabular dataset.

    Arguments:
        path (str): The path to the dataset to load.
        format (str): The format of the data file. One of "CSV", "TSV", or "JSON" (case-insensitive).
        fields tuple(str, str): The names of the columns in the format: (text feature, label).
        preprocessing: The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: None.
        tokenize: The function used to tokenize strings using this field into
            sequential examples. If "spacy", the SpaCy English tokenizer is
            used. Default: str.split.
        embedding_dim: The dimension for the sentence vector embeddings. Default: 60.
        batch_size: Batch_size. Default: 4.
        device: Device to create batches on. Use -1 for CPU and None for the currently active GPU device.
    """
    text = torchtext.data.Field(tokenize=tokenize, preprocessing=preprocessing, lower=True, batch_first=True)
    label = torchtext.data.Field(sequential=False, batch_first=True, unk_token=None)

    fields = {fields[0]: ('text', text), fields[1]: ('label', label)}

    dset = torchtext.data.TabularDataset(path, format, fields)

    text.build_vocab(dset, min_freq=2)
    label.build_vocab(dset)

    train, test, val = dset.split([.6, .01, .39])

    return torchtext.data.Iterator.splits((train, val, test), batch_size=batch_size, device=device)


def twitter(path='data/twitter_us_airline_sentiment.csv', fields=('text', 'airline_sentiment'), **kwargs):
    import nltk
    import dataloader.process_tweets as pt
    kwargs.setdefault('preprocessing', pt.preprocess())
    kwargs.setdefault('tokenize', nltk.tokenize.TweetTokenizer().tokenize)
    return lambda: load(path, format='csv', fields=fields, **kwargs)


def yelp(path='data/yelp_review.csv', **kwargs):
    kwargs.setdefault('tokenize', 'spacy')
    return lambda: load(path, format='csv', fields=('text', 'stars'), **kwargs)
