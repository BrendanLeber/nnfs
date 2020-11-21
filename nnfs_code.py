# -*- coding: utf-8 -*-

import numpy as np

import nnfs
from nnfs.datasets import spiral_data, vertical_data


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
    X, y = spiral_data(samples=100, classes=3)  # vertical_data(samples=100, classes=3)

    # create model
    dense1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs, 3 outputs
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs
    activation2 = Activation_Softmax()

    # create loss function
    loss_function = Loss_CategoricalCrossentropy()

    # helper variables
    lowest_loss = 999999  # some initial value
    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases = dense2.biases.copy

    for iteration in range(10000):
        # update weights with some small random values
        dense1.weights += 0.05 * np.random.randn(2, 3)
        dense1.biases += 0.05 * np.random.randn(1, 3)
        dense2.weights += 0.05 * np.random.randn(3, 3)
        dense2.biases += 0.05 * np.random.randn(1, 3)

        # perform a forward pass of our training data through this layer
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        # perform a forward pass through the activation function
        # it takes the output of the second dense layer here and returns loss
        loss = loss_function.calculate(activation2.output, y)

        # calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == y)

        # if loss is smaller - print and save weights and biases
        if loss < lowest_loss:
            print(
                f"New set of weights found, iteration: {iteration}, loss: {loss}, acc: {accuracy}"
            )
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            lowest_loss = loss
        else:  # revert weights and biases
            dense1.weights = best_dense1_weights.copy()
            dense1.biases = best_dense1_biases.copy()
            dense2.weights = best_dense2_weights.copy()
            dense2.biases = best_dense2_biases.copy()
