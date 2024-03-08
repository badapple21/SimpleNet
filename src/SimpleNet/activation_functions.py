import numpy as np
import warnings


def sigmoid(x):
    warnings.filterwarnings("ignore")
    return 1 / (1 + np.exp(-1 * np.float16(x)))


def fake_desigmoid(x):
    return x * (1 - x)


def relu(x):
    if x > 0:
        return x
    else:
        return 0


def leaky_relu(x):
    if x > 0:
        return x
    else:
        return 0.01 * x


def de_leaky_relu(x):
    if x < 0:
        return 0.01
    else:
        return 1


def tanh(x):
    return 2 / (1 + np.exp(-2 * x))
