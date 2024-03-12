import matrix_math
import activation_functions
import utils

# from . import utils
# from . import activation_functions
# from . import matrix_math

import numpy as np
import pickle
import time
import os
from rich.progress import track
from rich import print
import random


class NeuralNetwork:
    def __init__(
        self,
        input_nodes,
        hidden_nodes,
        output_nodes,
        activation_function,
        activation_function_derivative,
        bias_multiplier=1,
        weight_multiplier=0.1,
        learning_rate=0.1,
        learning_rate_decay=0.01,
    ):
        # num of neurons for each layer
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_nodes = hidden_nodes

        # PRESETS
        self.bias_multiplier = bias_multiplier
        self.weight_multiplier = weight_multiplier
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

        self.weights = []

        # loops through the hidden_nodes list to create a 2D list off all the weights
        for i in range(len(self.hidden_nodes) + 1):
            # checks if its the first hidden layer
            if i == 0:
                y = self.input_nodes
            else:
                y = self.hidden_nodes[i - 1]

            # checks if its the last layer
            if i == len(self.hidden_nodes):
                x = self.output_nodes
            else:
                x = self.hidden_nodes[i]

            # adds the matrix to the list randomizes the weights to a between -1 and 1 times the multiplier
            self.weights.append(matrix_math.matrix(x, y))
            self.weights[i].randomize(self.weight_multiplier)

        # loops through the num of hidden layers and creates a 1d matrix of the biases with a random value from -1 to 1 times the multiplier
        self.bias = []

        for i in range(len(self.hidden_nodes)):
            self.bias.append(matrix_math.matrix(hidden_nodes[i], 1))
            self.bias[i].randomize(self.bias_multiplier)

        # adds the bias matrix for the outputs
        self.bias.append(matrix_math.matrix(output_nodes, 1))
        self.bias[-1].randomize(self.bias_multiplier)

        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

        self.net = [self.weights, self.bias]

    def reset_net(self):
        for weights in self.net[0]:
            self.weight.randomize(self.weight_multiplier)
        for bias in self.net[1]:
            self.bias.randomize(self.bias_multiplier)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        if "[" in path:
            self.weights = data[0]
            self.bias = data[1]
        else:
            self.weights[0] = data[0]
            self.weights[1] = data[1]
            self.bias[0] = data[2]
            self.bias[1] = data[3]

    def save_net(self, path, file_name):
        if not os.path.exists(path):
            os.mkdir(path)

        with open(f"{path}/{file_name}.pickle", "wb") as f:
            pickle.dump(self.net, f)
            f.close()

    def feed_forward(self, input_array):
        # loads the input to a matrix
        inputs = matrix_math.from_array(input_array)

        # loops through every  hidden layer and feeds the inputs forward
        activations = [inputs]
        for i in range(len(self.weights)):
            # multiples the prevoius layers activations by the weights and then adds the bias then applies the activation function
            activations.append(matrix_math.multiply(self.weights[i], activations[i]))
            activations[i + 1].add(self.bias[i])
            activations[i + 1].map(self.activation_function)

        # returns the last layer activations as an array
        rtn = []
        for layer in activations:
            rtn.append(layer.to_array())
        return rtn

    def train(self, inputs_array, targets_array):
        outputs = self.feed_forward(inputs_array)

        targets = matrix_math.from_array(targets_array)

        layer_outputs = matrix_math.from_array(outputs[-1])

        errors = [matrix_math.subtract(targets, layer_outputs)]

        for i, weights in reversed(list(enumerate(self.weights[1:]))):
            weights_t = matrix_math.transpose(weights)
            error = matrix_math.multiply(weights_t, errors[0])

            errors.insert(0, error)

        for j, error in enumerate(errors):
            gradient = matrix_math.map(
                matrix_math.from_array(outputs[j + 1]),
                self.activation_function_derivative,
            )

            # gradient = matrix_math.from_array(outputs[j + 1])

            gradient.multiply(error)
            gradient.multiply(self.learning_rate)

            self.bias[j].add(gradient)

            output_t = matrix_math.from_array(outputs[j])
            output_t = matrix_math.transpose(output_t)
            delta = matrix_math.multiply(gradient, output_t)

            self.weights[j].add(delta)

    def test_net(self, test_images, test_labels):

        correct = 0
        total = 0
        for i in track(range(len(test_images)), description="[green]Testing: "):
            output = self.feed_forward(test_images[i])
            if utils.get_max(output[-1]) == test_labels[i]:
                correct += 1
            total += 1

        return (correct / total) * 100

    def test_and_train(
        self, test_images, test_labels, train_images, train_labels, iterations
    ):
        test_accuracys = []
        start_time = time.time()
        os.system("cls")

        for j in range(iterations):
            for i, image in enumerate(
                track(
                    train_images,
                    description=f"[green]Training Epoch {j+1}/{iterations}: ",
                )
            ):
                self.train(image, utils.get_correct(train_labels[i]))
                current_time = time.time()

            test_accuracys.append(round(self.test_net(test_images, test_labels), 2))

        return test_accuracys


def main():

    images, labels, test_images, test_labels = utils.load_MNIST()
    hidden_layers = [16, 16]

    network = NeuralNetwork(
        784,
        hidden_layers,
        10,
        activation_functions.sigmoid,
        activation_functions.fake_desigmoid,
        learning_rate=0.03,
    )  # creates network
    accuracy = network.test_and_train(test_images, test_labels, images, labels, 2)

    print(accuracy)


if __name__ == "__main__":
    main()
