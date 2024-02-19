import simpleNet as sn

images, labels, test_images, test_labels = sn.load_MNIST()
hiden_layers = [10]


network = sn.NeuralNetwork(784, hiden_layers, 10, sn.sigmoid)  # creates network
accuracy = network.test_and_train(test_images, test_labels, images, labels, 1)

print(accuracy)
network.save_net(f"saved_nets/{hiden_layers} {accuracy[0]}")
