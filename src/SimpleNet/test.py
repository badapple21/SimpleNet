# feeds the input through the net using the feed forward function
        activations = self.feed_forward(inputs_array)

        # Calculates error of last layer
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

            # # # calculates the gradient by  multiplying the activation of the layer by the error of the next layer times the learning rate
            # # gradients = activations[len(activations) - (i + 1)]
            # # gradients = matrix_math.from_array(gradients)

            # # multiply the error with the derivative of the activation function
            # last_layer = matrix_math.from_array(activations[-1*i-1])

            # gradients = matrix_math.map(last_layer, self.activation_function_derivative)

            # gradients.multiply(errors[i])
            # gradients.multiply(self.learning_rate)




            # Calculate the gradient by multiplying the derivative of the activation function
            # with the error of the last layer and the learning rate
            last_layer = matrix_math.from_array(activations[-1 - i])
            gradients = matrix_math.map(last_layer, self.activation_function_derivative)
            gradients.multiply(errors[i])
            gradients.multiply(self.learning_rate)


            # calculates the deltas by multiplying the gradients by the weight
            activations_t = matrix_math.transpose(
                matrix_math.from_array(activations[len(activations) - (i + 2)])
            )
            weight_deltas = matrix_math.multiply(gradients, activations_t)

            # adds the deltas and the gradients to the weights and bias
            self.weights[len(self.weights) - (1 + i)].add((weight_deltas))