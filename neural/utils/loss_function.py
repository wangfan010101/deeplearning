# loss function

import numpy as np


# mean squared error function
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# cross entropy error function
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    delta = 1e-7
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


def function_x1(x):
    return np.sum(x**2)


if __name__ == '__main__':
    t1 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    # print(cross_entropy_error(np.array(y1), np.array(t1)))
    print(numerical_gradient(function_x1, np.array([3.0, 0.0])))
