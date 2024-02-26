from . import nn 

images, labels, test_images, test_labels = nn.load_MNIST()
hidden_layers = [1]


network = nn.NeuralNetwork(784, hidden_layers, 10, nn.sigmoid)  # creates network
accuracy = network.test_and_train(test_images, test_labels, images, labels, 1)

print(accuracy)
network.save_net(f"src\SimpleNet\saved_nets\{hidden_layers} {accuracy[0]}")
