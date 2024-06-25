import numpy as np


class Sigmoid:
    def forward(self, x):
        self.input = x
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, d_output):
        sigmoid_derivative = self.output * (1 - self.output)
        return d_output * sigmoid_derivative
