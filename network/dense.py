import numpy as np


class Dense:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))
        self.activation = activation
    
    def forward(self, input):
        self.input = input
        self.z = np.dot(input, self.weights) + self.biases
        self.output = self.z if self.activation is None else self.activation.forward(self.z)
        return self.output
    
    def backward(self, d_output, learning_rate):
        if self.activation:
            d_output = self.activation.backward(d_output)
        
        d_input = np.dot(d_output, self.weights.T)
        d_weights = np.dot(self.input.T, d_output)
        d_biases = np.sum(d_output, axis=0, keepdims=True)
        
        # Update parameters
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        
        return d_input