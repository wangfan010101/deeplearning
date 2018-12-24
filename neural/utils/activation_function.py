# activation function
import numpy as np


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# RELU function
def relu(x):
    return np.maximum(0, x)


# softmax function
def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y


if __name__ == "__main__":
    pass
