# -*- coding: utf-8 -*-

import numpy as np

import nnfs
from nnfs.datasets import spiral_data


nnfs.init()


class Layer_Dense:
    def __init__(self, inputs, neurons) -> None:
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    def forward(self, inputs) -> None:
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues) -> None:
        # gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs) -> None:
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues) -> None:
        # make a copy since we need to modify the original values
        self.dinputs = dvalues.copy()
        # zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs) -> None:
        self.inputs = inputs

        # get unnormalized probablities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues) -> None:
        self.dinputs = np.empty_like(dvalues)

        # enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # flatten output array
            single_output = single_output.reshape(-1, 1)
            # calculate jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    """Common loss class."""

    def calculate(self, output, y) -> float:
        """Calculates the data and regularization losses
        given model output and ground truth values.
        """

        # calculate sample losses
        sample_losses = self.forward(output, y)

        # calculate mean loss
        data_loss = np.mean(sample_losses)

        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    """Cross-entropy loss."""

    def forward(self, y_pred, y_true) -> float:
        # number of samples in a batch
        samples = len(y_pred)

        # clip data to prevent division by zero.
        # clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # probabilities for target values only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped ** y_true, axis=1)

        # losses
        negative_log_likelyhoods = -np.log(correct_confidences)

        return negative_log_likelyhoods

    def backward(self, dvalues, y_true) -> None:
        samples = len(dvalues)
        labels = len(dvalues[0])

        # if labels are sparse, turn into a one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # calculate gradient
        self.dinputs = -y_true / dvalues
        # normalize gradient
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy:
    """Softmax classifier.

    Combined Softmax activation and cross-entropy loss for faster backward step.
    """

    def __init__(self):
        """Creates activation and loss function objects."""
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true) -> float:
        # output layers activation function
        self.activation.forward(inputs)
        # set the output
        self.output = self.activation.output
        # calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true) -> None:
        samples = len(dvalues)

        # if labels are one-hot encoded turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # copy so we can safely modify
        self.dinputs = dvalues.copy()
        # calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # normalize gradient
        self.dinputs = self.dinputs / samples


class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.current_learning_rate = self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    def pre_update_params(self) -> None:
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer) -> None:
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = (
                self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            )
            layer.weight_momentums = weight_updates

            bias_updates = (
                self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            )
            layer.bias_momentums = bias_updates
        else:
            weight_updates += -self.current_learning_rate * layer.dweights
            bias_updates += -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self) -> None:
        self.iterations += 1


class Optimizer_Adagrad:
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        self.current_learning_rate = self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0

    def pre_update_params(self) -> None:
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer) -> None:
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        layer.weights += (
            -self.current_learning_rate
            * layer.dweights
            / (np.sqrt(layer.weight_cache) + self.epsilon)
        )
        layer.biases += (
            -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        )

    def post_update_params(self) -> None:
        self.iterations += 1


class Optimizer_RMSprop:
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
        self.current_learning_rate = self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho
        self.iterations = 1

    def pre_update_params(self) -> None:
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer) -> None:
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        layer.weights += (
            -self.current_learning_rate
            * layer.dweights
            / (np.sqrt(layer.weight_cache) + self.epsilon)
        )
        layer.biases += (
            -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        )

    def post_update_params(self) -> None:
        self.iterations += 1


class Optimize_Adam:
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.current_learning_rate = self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iterations = 0

    def pre_update_params(self) -> None:
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer) -> None:
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = (
            self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        )
        layer.bias_momentums = (
            self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        )

        weight_momentums_corrected = layer.weight_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = (
            self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        )
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += (
            -self.current_learning_rate
            * weight_momentums_corrected
            / (np.sqrt(weight_cache_corrected) + self.epsilon)
        )
        layer.biases += (
            -self.current_learning_rate
            * bias_momentums_corrected
            / (np.sqrt(bias_cache_corrected) + self.epsilon)
        )

    def post_update_params(self) -> None:
        self.iterations += 1


if __name__ == "__main__":
    X, y = spiral_data(samples=100, classes=3)

    dense1 = Layer_Dense(2, 64)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    # optimizer = Optimizer_SGD(decay=8e-8, momentum=0.9)
    # optimizer = Optimizer_Adagrad(decay=1e-4)
    # optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)
    optimizer = Optimize_Adam(learning_rate=0.05, decay=5e-7)

    # train in a loop
    for epoch in range(10001):
        # forward pass
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)

        # calculate accuracy from output of activation2 and targets
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)

        if not epoch % 100:
            print(
                (
                    f"epoch: {epoch}"
                    f", acc: {accuracy:.3f}"
                    f", loss: {loss:.3f}"
                    f", lr: {optimizer.current_learning_rate}"
                )
            )

        # backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
