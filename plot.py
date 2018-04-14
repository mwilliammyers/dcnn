import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('fp', metavar='FILE', nargs='+', help='Path to JSON file containing data')
parser.add_argument(
    '-v',
    '--val-only',
    dest='val_only',
    action='store_true',
    default=False,
    help='Show only the validation loss/accuracy')
args = parser.parse_args()

files = args.fp
val_only = args.val_only
title = "plot"

data = defaultdict(lambda: [None for i in range(len(files))])
for i, f in enumerate(files):
    d = json.load(open(f))
    if title == 'plot':
        title = re.sub(r'run(-|_)?\d+', 'avg', d['title'])
    for k, v in d.items():
        if k.startswith("stats"):
            data[k][i] = np.array([p[2] for p in v])

avgeraged_data = defaultdict(list)
for k, v in data.items():
    for x in zip(*v):
        avgeraged_data[k].append(np.mean(x))

# k = 'stats/train_loss'
# print(k, data[k][0][0], data[k][1][0], data[k][2][0], data[k][3][0])
# x = (data[k][0][0] + data[k][1][0] + data[k][2][0] + data[k][3][0]) / 4.
# print(k, data[k][0][1], data[k][1][1], data[k][2][1], data[k][3][1])
# y = (data[k][0][1] + data[k][1][1] + data[k][2][1] + data[k][3][1]) / 4.
#
# print(avgeraged_data['stats/train_loss'][0], x, avgeraged_data['stats/train_loss'][1], y)
#
# raise

try:
    import seaborn as sns
    sns.set()
    sns.set_style('darkgrid', {'axes.facecolor': '.88'})
    sns.set_context('paper')
except ImportError:
    pass

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4, 2.75))
for k, v in avgeraged_data.items():
    label = k.split('/')[-1].replace('_', ' ')
    if val_only and 'train' in label:
        continue
    (ax1 if 'loss' in label else ax2).plot(v, label=label)

# plt.title(title)
ax1.legend()
ax1.set_ylabel('Loss')
ax2.legend(loc='lower right')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Accuracy')
fig.tight_layout()

plt.savefig(f'figures/{title}.png')
