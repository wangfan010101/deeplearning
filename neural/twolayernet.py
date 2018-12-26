# This is a demo of two layer net

import numpy as np
import sys
import os
sys.path.append(os.path.pardir)
from neural.utils import activation_function
from neural.utils import loss_function
from neural import demo


class TwoLayerNet(object):

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        w1, w2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, w1) + b1
        z1 = activation_function.sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = activation_function.softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return loss_function.cross_entropy_error(y, t)

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)

        grads = dict()
        grads['W1'] = demo.numerical_gradient(loss_w, self.params['W1'])
        grads['b1'] = demo.numerical_gradient(loss_w, self.params['b1'])
        grads['W2'] = demo.numerical_gradient(loss_w, self.params['W2'])
        grads['b2'] = demo.numerical_gradient(loss_w, self.params['b2'])

        return grads


if __name__ == '__main__':
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

