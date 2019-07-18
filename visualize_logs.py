import numpy as np
import torch
import matplotlib.pyplot as plt
import os

accs = np.zeros(76)
for i in range(1,77):
    acc = np.load("logs/accLog" + str(i) + ".npy")
    accs[i-1] = acc

#print(accs)

plt.plot(accs)
plt.show()

path ="/Users/sophia/server/logsLowLR"


val_loss = np.zeros(109)
val_acc = np.zeros(109)
for i in range(1,110):
    val_loss[i-1] = np.load(os.path.join(path, "lossLog" + str(i) + ".npy"))
    val_acc[i-1] = np.load(os.path.join(path, "accLog" + str(i) + ".npy"))

plt.plot(val_loss)
plt.show()
plt.plot(val_acc)
plt.show()

train_loss = np.zeros(109)
train_acc = np.zeros(109)
for i in range(1, 110):
    with open(os.path.join(path, "train_loss" + str(i) + ".txt")) as f:
        train_loss[i-1] = np.array(f.read()).astype(np.float)
    with open(os.path.join(path, "train_acc" + str(i) + ".txt")) as f:
        train_acc[i - 1] = np.array(f.read()).astype(np.float)


plt.plot(train_loss)
plt.show()
plt.plot(train_acc)
plt.show()
