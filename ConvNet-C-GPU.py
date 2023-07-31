# coding: utf-8
import cupy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
from convnet import ConvNet
from optimizer import *
import pickle

# 0: Load MNIST data
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# x_train, t_train = x_train[:5000], t_train[:5000]
# x_test, t_test = x_test[:1000], t_test[:1000]

# 1: Settings
network = ConvNet(input_dim=(1, 28, 28),
                  conv_param={'filter_num': 32, 'filter_size': 5, 'pad': 2, 'stride': 1},
                  hidden_size=100, output_size=10, weight_init_std=0.01)
optimizer = Adam()
max_epochs = 10
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = max(train_size / batch_size, 1)
max_iter = int(max_epochs * iter_per_epoch)

train_loss_list = []
train_acc_list = []
test_acc_list = []

epoch_cnt = 0

for i in range(max_iter):
    batch_mask = np.random.choice(train_size, batch_size).get()
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(np.asarray(x_batch), np.asarray(t_batch))
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(np.asarray(x_train), np.asarray(t_train))
        test_acc = network.accuracy(np.asarray(x_test), np.asarray(t_test))
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

# 2: Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(network, f)

# 3: Plot
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x.get(), np.asarray(train_acc_list), marker='o', label='train', markevery=2)
plt.plot(x.get(), np.asarray(test_acc_list), marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
