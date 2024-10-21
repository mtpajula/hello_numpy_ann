import numpy as np


# Define a perceptron (single neuron) layer
class Perceptron:
    def __init__(self, input_size, activation=None):
        # Initialize weights and bias
        self.weights = np.random.randn(input_size) * 0.1
        self.bias = 0.0
        self.activation = activation
        
    def forward(self, input):
        # Store input for use in backward pass
        self.input = input
        # Linear transformation
        z = np.dot(input, self.weights) + self.bias
        # Apply activation function if any
        self.output = z if self.activation is None else self.activation.forward(z)
        return self.output
    
    def backward(self, d_output, learning_rate):
        # If activation function is used, apply its derivative
        if self.activation:
            d_output = d_output * self.activation.backward(self.output)

        # Reshape d_output to (batch_size, 1) for broadcasting
        d_output = d_output.reshape(-1, 1)  # Shape: (batch_size, 1)

        # Calculate gradients
        d_input = d_output * self.weights
        d_weights = d_output * self.input
        d_bias = d_output
        
        # Sum over the batch to get the total gradients
        d_weights = np.sum(d_weights, axis=0)  # Shape: (input_size,)
        d_bias = np.sum(d_bias, axis=0)  # Shape: (1,)

        # Update parameters
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias[0]  # Extract scalar from array
        
        return d_input
