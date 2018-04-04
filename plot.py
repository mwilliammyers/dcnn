import matplotlib.pyplot as plt
import numpy as np
import struct
import os

base = './logs'
fp = base + '/stats'
data = open(fp, 'rb').read()
data = struct.unpack('f' * (len(data) // 4), data)
data = np.array(data).reshape(-1, 4)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(data[:,0], label='train')
ax1.plot(data[:,2], label='val')
ax1.legend()
ax1.set_title('Loss')
ax1.set_ylabel('Loss (cross entropy)')

ax2.plot(data[:,1] * 100, label='train')
ax2.plot(data[:,3] * 100, label='val')
ax2.legend()
ax2.set_title('Accuracy')
ax2.set_xlabel('Batches')
ax2.set_ylabel('Accuracy')

plt.show()
