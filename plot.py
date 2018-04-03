import matplotlib.pyplot as plt
import os

# base = os.path.dirname(os.path.realpath(__file__))
base = './logs'
train_file = base + '/train_loss'
val_file = base + '/val_loss'
train_loss = [float(x) for x in open(train_file).read().strip().split('\n')]
val_loss = [float(x) for x in open(val_file).read().strip().split('\n')]

plt.plot(train_loss, label='train')
plt.plot(val_loss, label='val')

plt.xlabel('Iterations * 50')
plt.ylabel('Loss (cross entropy)')
plt.legend()

plt.show()
