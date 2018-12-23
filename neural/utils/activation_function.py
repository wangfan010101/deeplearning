# activation function
import numpy as np


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def relu(x):
	return np.maximum(0, x)


def softmax(x):
	exp_x = np.exp(x)
	sum_exp_x = np.sum(exp_x)
	y = exp_x / sum_exp_x
	return y


print(softmax(np.array([0.1,0.2,0.3,0.4,0.5])))