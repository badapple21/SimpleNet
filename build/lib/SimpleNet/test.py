import nn
from activation_functions import sigmoid
from utils import load_MNIST


images, labels, test_images, test_labels = load_MNIST()
hidden_layers = [1]


network = nn.NeuralNetwork(784, hidden_layers, 10, sigmoid)  # creates network
accuracy = network.test_and_train(test_images, test_labels, images, labels, 1)

print(accuracy)
network.save_net("src\SimpleNet\saved_nets2", f"{hidden_layers}_{accuracy[0]}")
