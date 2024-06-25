import numpy as np


# Define a dense (fully connected) layer
class Dense:
    def __init__(self, input_size, output_size, activation=None):
        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))
        self.activation = activation
    
    def forward(self, input):
        # Store input for use in backward pass
        self.input = input
        # Linear transformation
        self.z = np.dot(input, self.weights) + self.biases
        # Apply activation function if any
        self.output = self.z if self.activation is None else self.activation.forward(self.z)
        return self.output
    
    def backward(self, d_output, learning_rate):
        # Apply activation function's backward pass if any
        if self.activation:
            d_output = self.activation.backward(d_output)
        
        # Calculate gradients
        d_input = np.dot(d_output, self.weights.T)
        d_weights = np.dot(self.input.T, d_output)
        d_biases = np.sum(d_output, axis=0, keepdims=True)
        
        # Update parameters
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        
        return d_input