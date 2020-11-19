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
