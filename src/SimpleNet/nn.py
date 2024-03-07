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


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, activation_function):
        # num of neurons for each layer
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_nodes = hidden_nodes
        self.bias_multiplier = 10
        self.weight_multiplier = 0.1

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
            self.weights[i].randomize(0.1)

        # loops through the num of hidden layers and creates a 1d matrix of the biases with a random value from -1 to 1 times the multiplier
        self.bias = []

        for i in range(len(self.hidden_nodes)):
            self.bias.append(matrix_math.matrix(hidden_nodes[i], 1))
            self.bias[i].randomize(10)

        # adds the bias matrix for the outputs
        self.bias.append(matrix_math.matrix(output_nodes, 1))
        self.bias[-1].randomize(10)

        # sets learning rate
        self.learning_rate = 0.3

        self.activation_function = activation_function

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
        # feeds the input through the net using the feed forward function
        activations = self.feed_forward(inputs_array)

        errors = [
            matrix_math.subtract(
                matrix_math.from_array(targets_array),
                matrix_math.from_array(activations[-1]),
            )
        ]

        # calculates the gradients and bias for each layer then it adds the gradients to the current layer
        for i in range(len(activations) - 1):
            # subtracts the targets and outputs to get the error
            errors.append(
                matrix_math.multiply(
                    matrix_math.transpose(self.weights[len(activations) - (i + 2)]),
                    errors[i],
                )
            )

            # calculates the gradient by  multiplying the activation of the layer by the error of the next layer times the learning rate
            gradients = activations[len(activations) - (i + 1)]
            gradients = matrix_math.from_array(gradients)
            gradients.multiply(errors[i])
            gradients.multiply(self.learning_rate)

            # calculates the deltas by multiplying the gradients by the weight
            activations_t = matrix_math.transpose(
                matrix_math.from_array(activations[len(activations) - (i + 2)])
            )
            weight_deltas = matrix_math.multiply(gradients, activations_t)

            # adds the deltas and the gradients to the weights and bias
            self.weights[len(self.weights) - (1 + i)].add((weight_deltas))

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
    images, labels, test_images, test_labels = load_MNIST()

    hidden_layers = [1, 1]

    network = NeuralNetwork(
        784, hidden_layers, 10, activation_functions.sigmoid
    )  # creates network
    accuracy = network.test_and_train(test_images, test_labels, images, labels, 2)

    print(accuracy)
    network.save_net(f"src\SimpleNet\saved_nets\{hidden_layers} {accuracy[0]}")


if __name__ == "__main__":
    main()
