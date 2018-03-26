from torchtext import data as ttd
import process_tweets as pt
import nltk

tokenizer = nltk.tokenize.TweetTokenizer()
text = ttd.Field(
	tokenize=tokenizer.tokenize, 
	preprocessing=pt.preprocess(), 
	lower=True, 
	batch_first=True)
label = ttd.Field(sequential=False, batch_first=True)

fields = {'airline_sentiment': ('label',label), 'text': ('text', text)}
dset = ttd.TabularDataset('../data/twitter_airlines.csv', 'csv', fields)

text.build_vocab(dset)
label.build_vocab(dset)

print (text.vocab.freqs)