# neural network

import numpy as np
import sys
import os
import gzip
import urllib.request
import pickle
sys.path.append(os.pardir)


url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}


def downloads():
    for file in key_file.values():
        if os.path.exists('mnist/' + file):
            print(file + ' is alread exists, continue')
            continue
        print("begin to download " + file)
        urllib.request.urlretrieve(url_base + file, 'mnist/' + file)
        print("download finished " + file)


def load_label(file_name):
    file_path = 'mnist/' + file_name
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    return labels


def load_image(file_name):
    file_path = 'mnist/' + file_name
    with gzip.open(file_path, 'rb') as f:
        imgs = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        imgs = imgs.reshape(-1, 784)

    return imgs


def convert_numpy():
    dataset = dict()
    dataset['train_img'] = load_image(key_file['train_img'])
    dataset['train_label'] = load_label(key_file['train_label'])
    dataset['test_img'] = load_image(key_file['test_img'])
    dataset['test_label'] = load_label(key_file['test_label'])

    return dataset


def change_one_hot_label(x):
    t = np.zeros((x.size, 10))
    for idx, row in enumerate(t):
        row[x[idx]] = 1

    return t


def init_mnist():
    downloads()
    dataset = convert_numpy()
    with open('mnist/mnist.pkl', 'wb') as f:
        pickle.dump(dataset, f, -1)
    print('Done!')


def main(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists('mnist/mnist.pkl'):
        init_mnist()

    with open('mnist/mnist.pkl', 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
    init_mnist()
