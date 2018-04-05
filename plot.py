import matplotlib.pyplot as plt
import numpy as np
import struct
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '-f',
    dest='fp',
    metavar='FILE',
    default='logs/stats',
    help='Path to binary file containing data')
args = parser.parse_args()

fp = args.fp
data = open(fp, 'rb').read()
data = struct.unpack('f' * (len(data) // 4), data)
data = np.array(data).reshape(-1, 4)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(data[:,0], color='b', label='train')
ax1.plot(data[:,2], color='g', label='val')
ax1.legend()
ax1.set_title('Loss')
ax1.set_ylabel('Loss (cross entropy)')

data[:,[1,3]] *= 100
ax2.plot(data[:,1], color='b', label='train')
ax2.plot(data[:,3], color='g', label='val')
ax2.hlines(
    (data[:,1].max(), data[:,3].max()),
    0, len(data)-1,
    colors=('b', 'g'),
    linestyle='dotted')
ax2.legend(loc='lower right')
ax2.set_title('Accuracy')
ax2.set_xlabel('Batches')
ax2.set_ylabel('Accuracy')

plt.show()
