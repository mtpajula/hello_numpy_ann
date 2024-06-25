import numpy as np


# Define the Sigmoid activation function
class Sigmoid:
    def forward(self, x):
        # Store input for use in backward pass
        self.input = x
        # Apply Sigmoid activation
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, d_output):
        # Calculate gradient for Sigmoid
        sigmoid_derivative = self.output * (1 - self.output)
        return d_output * sigmoid_derivative
