import numpy as np


# Define the ReLU activation function
class ReLU:
    def forward(self, x):
        # Store input for use in backward pass
        self.input = x
        # Apply ReLU (Rectified Linear Unit) activation
        return np.maximum(0, x)
    
    def backward(self, d_output):
        # Calculate gradient for ReLU
        d_input = d_output.copy()
        d_input[self.input <= 0] = 0
        return d_input