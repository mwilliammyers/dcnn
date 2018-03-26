import pandas
import re

def strip_emoticons(text):
	# import pdb; pdb.set_trace()
	if type(text) in (tuple, list):
		chars = set(' '.join(text))
	else:
		chars = set(text)
		text = [text]
	emoticons = ''.join([x for x in chars if ord(x) > 1000])
	if len(emoticons) > 0:
		exp = re.compile('[{}]'.format(emoticons))
		text = [exp.sub(' ', t) for t in text]
	return text

def replace_urls(text, token='URL'):
	# NOTE: this method will remove anything that comes after the url and
	# before a whitespace character along with the url. Maybe not the desired
	# behaviour?
	if type(text) == str:
		text = [text]
	exp = re.compile(r'https?://\S+')
	text = [exp.sub(token, t) for t in text]
	return text

def replace_names(text, token='USERNAME'):
	if type(text) == str:
		text = [text]
	exp = re.compile('@\w*')
	text = [exp.sub(token, t) for t in text]
	return text

def replace_duplicate_letters(text):
	if type(text) == str:
		text = [text]
	exp = re.compile(r'((.)\2{2,})')
	text = [exp.sub(lambda match: match.groups()[1]*2, t) for t in text]
	return text

def preprocess():

	def proc(text):
		text = strip_emoticons(text)
		text = replace_names(text)
		text = replace_urls(text)
		text = replace_duplicate_letters(text)
		return text
	return proc


def process_tweets(fpath='data/twitter_airlines.csv'):

	cols = ['text', 'airline_sentiment']
	data = pandas.read_csv(fpath)[cols]
	split_token = '\n:\n'
	text = split_token.join(list(data['text'])).lower()
	text = strip_emoticons(text)
	text = replace_names(text)
	text = replace_urls(text)
	text = replace_duplicate_letters(text)
	data['text'] = text.split(split_token)
	return list(zip(*[list(data[key]) for key in cols]))



if __name__ == '__main__':
	data = process_tweets()
	print(data)