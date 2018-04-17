import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    'fp',
    metavar='FILE',
    nargs='+',
    help='Path to JSON file containing data')  # yapf: disable
parser.add_argument(
    '-v',
    '--val-only',
    dest='val_only',
    action='store_true',
    default=False,
    help='Show only the validation loss/accuracy')  # yapf: disable
parser.add_argument(
    '-m',
    '--mean',
    dest='compute_mean',
    action='store_true',
    default=False,
    help='Compute the mean of multiple runs')  # yapf: disable
parser.add_argument(
    '-s',
    '--fig-size',
    dest='fig_size',
    metavar='SIZE',
    type=float,
    nargs='*',
    default=[4, 2.75],
    help='Figure size')  # yapf: disable
parser.add_argument(
    '--legend-font-size',
    dest='legend_font_size',
    metavar='SIZE',
    type=float,
    default=9,
    help='Legend font size')  # yapf: disable
parser.add_argument(
    '--label-font-size',
    dest='label_font_size',
    metavar='SIZE',
    type=float,
    help='Label font size')  # yapf: disable

args = parser.parse_args()
if not args.label_font_size:
    args.label_font_size = args.legend_font_size + 1.5

data = defaultdict(lambda: [None for i in range(len(args.fp))] if args.compute_mean else list)
for i, f in enumerate(args.fp):
    d = json.load(open(f))
    for k, v in d.items():
        if k.startswith("stats"):
            if args.compute_mean:
                data[k][i] = np.array([p[2] for p in v])
            else:
                data[k.replace('/', f"/{d['title'].split('_')[0]}_")] = [p[2] for p in v]

if args.compute_mean:
    plot_data = defaultdict(list)
    for k, v in data.items():
        for x in zip(*v):
            plot_data[k].append(np.mean(x))
else:
    plot_data = data

try:
    import seaborn as sns
    sns.set()
    sns.set_style('darkgrid', {'axes.facecolor': '.88'})
    sns.set_context('paper')
except ImportError:
    pass

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=args.fig_size)
for key, trace in plot_data.items():
    label = re.sub('-|_', ' ', key.split('/')[-1])
    if args.val_only and 'train' in label:
        continue
    (ax1 if 'loss' in label else ax2).plot(trace, label=label)

ax1.legend(loc='upper right', fontsize=args.legend_font_size)
ax1.set_ylabel('Loss', fontsize=args.label_font_size)
ax2.legend(loc='lower right', fontsize=args.legend_font_size)
ax2.set_xlabel('Iterations', fontsize=args.label_font_size)
ax2.set_ylabel('Accuracy', fontsize=args.label_font_size)
fig.tight_layout()

# plt.show()
plt.savefig(str(Path.home().joinpath('Downloads').joinpath('figure.png')))
# plt.savefig(f"figures/{title}{'_val' if args.val_only else ''}.png")
