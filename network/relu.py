import numpy as np


class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, d_output):
        d_input = d_output.copy()
        d_input[self.input <= 0] = 0
        return d_input
