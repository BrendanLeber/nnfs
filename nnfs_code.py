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


if __name__ == "__main__":
    # create dataset
    X, y = spiral_data(samples=100, classes=3)  # vertical_data(samples=100, classes=3)

    # create model
    dense1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs, 3 outputs
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs

    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    print(loss_activation.output[:5])
    print(f"loss: {loss}")

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    print(f"accuracy: {accuracy}")

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    print(dense1.dweights)
    print(dense1.dbiases)
    print(dense2.dweights)
    print(dense2.dbiases)
