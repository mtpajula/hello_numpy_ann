import numpy as np


# Define the Softmax activation function
class Softmax:
    def forward(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output
    
    def backward(self, d_output):
        return d_output
