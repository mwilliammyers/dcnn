import torch
from torchtext import data as ttd


def load(embedding_dim, batch_size, device=0, file_path='/mnt/pccfs/not_backed_up/data/yelp/yelp_review_small.csv'):
    """Load the moview review sentiment dataset.

    Arguments:
        embedding_dim: The dimension for the sentence vector embeddings.
        batch_size: Batch_size
        device: Device to create batches on. Use - 1 for CPU and None for the currently active GPU device.
    """
    text = ttd.Field(tokenize="spacy", lower=True, batch_first=True)
    label = ttd.Field(sequential=False, batch_first=True, unk_token=None)

    fields = {'stars': ('label', label), 'text': ('text', text)}
    dset = ttd.TabularDataset(file_path, 'csv', fields)

    text.build_vocab(dset)
    label.build_vocab(dset)

    embed_vector = torch.rand(len(text.vocab), embedding_dim)
    text.vocab.set_vectors(text.vocab.stoi, embed_vector, embedding_dim)

    train, test, val = dset.split([.6, .2, .2])

    return ttd.Iterator.splits((train, val, test), batch_size=batch_size, device=device)
