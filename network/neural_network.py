import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, d_output, learning_rate):
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output, learning_rate)
    
    def fit(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss (mean squared error)
            loss = np.mean((output - y) ** 2)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
            
            # Backward pass
            d_output = 2 * (output - y) / y.size
            self.backward(d_output, learning_rate)

    def predict(self, X):
        return self.forward(X)
