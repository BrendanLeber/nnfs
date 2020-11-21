# -*- coding: utf-8 -*-

import numpy as np
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()


class Activation_ReLU:
    def forward(self, inputs) -> None:
        """Calculate output values from inputs."""
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs) -> None:
        # get unnormalized probablities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        """Initialize weights and biases."""
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs) -> None:
        """Calculate output values from inputs, weights, and biases."""
        self.output = np.dot(inputs, self.weights) + self.biases


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


if __name__ == "__main__":
    # create dataset
    X, y = spiral_data(samples=100, classes=3)

    # create dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 3)

    # create ReLU activation (to be used with dense layer)
    activation1 = Activation_ReLU()

    # create second dense layer with 3 input features (as we take output
    # of previous layer here) and 3 output values
    dense2 = Layer_Dense(3, 3)

    # create Softmax activation (to be used with dense layer)
    activation2 = Activation_Softmax()

    # create a loss function
    loss_function = Loss_CategoricalCrossentropy()

    # perform a forward pass of our training data through this layer
    dense1.forward(X)

    # make a forward pass through activation function
    # it takes the output of the first dense layer here
    activation1.forward(dense1.output)

    # make a forward pass through second dense layer
    # it takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)

    # make a forward pass through activation function
    # it takes outputs of second dense layer here
    activation2.forward(dense2.output)

    # see the output of the first few samples
    print(activation2.output[:5])

    # perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = loss_function.calculate(activation2.output, y)

    # print loss value
    print(f"loss: {loss}")

    # calculate accuracy from output of activation2 and targets
    # calculate along first axis
    predictions = np.argmax(activation2.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    # print accuracy
    print(f"accuracy: {accuracy}")
