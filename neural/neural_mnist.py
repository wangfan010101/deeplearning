# a test for mnist

import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neural import mnist
from neural import twolayernet


def get_data():
    (train_img, train_label), (test_img, test_label) \
        = mnist.main(normalize=True, flatten=True, one_hot_label=True)

    return (train_img, train_label), (test_img, test_label)


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = get_data()
    train_loss_list = []

    # 超参数
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    network = twolayernet.TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    count = 0
    for i in range(iters_num):
        count += 1
        print("迭代次数: %s" % count)
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.numerical_gradient(x_batch, t_batch)

        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        print(train_loss_list)
