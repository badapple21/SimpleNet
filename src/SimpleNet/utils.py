import mnist
from mnist import MNIST
import os


def get_max(x):
    index = 0
    max = x[0]
    for i in range(len(x)):
        if x[i] > max:
            max = x[i]
            index = i

    return index


def load_MNIST():

    print(os.getcwd())
    mndata = MNIST("samples")
    images, labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    return images, labels, test_images, test_labels


def get_correct(label):
    x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    x[label] = 1
    return x
