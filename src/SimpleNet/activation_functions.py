import numpy as np
import warnings


def sigmoid(x):
    warnings.filterwarnings("ignore")
    return 1 / (1 + np.exp(-1 * np.float16(x)))


def fake_desigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    if x > 0:
        return x
    else:
        return 0


def leaky_relu(x):
    if x > 0:
        return x
    else:
        return 0.001 * x


def de_leaky_relu(x):
    if x > 0:
        return 1
    else:
        return 0.001


def tanh(x):
    return 2 / (1 + np.exp(-2 * x))

def soft_max(x):
    pass

def arg_max(x): 
    pass