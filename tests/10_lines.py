import SimpleNet as sn
from pathlib import Path

images, labels, test_images, test_labels = sn.load_MNIST()
hidden_layers = [1]


network = sn.NeuralNetwork(784, hidden_layers, 10, sn.sigmoid)  # creates network
accuracy = network.test_and_train(test_images, test_labels, images, labels, 5)

print(accuracy)

file_path = Path(__file__)
file_dir = file_path.parent
network.save_net(f"{file_dir}\saved_nets\{hidden_layers} {accuracy[0]}")
