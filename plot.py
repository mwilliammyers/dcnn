import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import struct
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('fp', metavar='FILE', nargs='+', help='Path to binary file containing data')
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
data = defaultdict(list)
for f in files:
    d = json.load(open(f))
    for k, v in d.items():
        if k.startswith("stats"):
            data[k].extend([p[2] for p in v])

try:
    import seaborn as sns
    sns.set()
    sns.set_style('darkgrid', {'axes.facecolor': '.88'})
    sns.set_context('paper')
except ImportError:
    pass

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4, 2.75))
for k, v in data.items():
    label = k.split('/')[-1].replace('_', ' ')
    print(label)
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
