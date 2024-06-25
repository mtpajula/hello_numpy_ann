import numpy as np


# Define the neural network
class NeuralNetwork:
    def __init__(self):
        # Initialize list of layers
        self.layers = []
    
    def add(self, layer):
        # Add a layer to the network
        self.layers.append(layer)
    
    def forward(self, input):
        # Forward pass through all layers
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, d_output, learning_rate):
        # Backward pass through all layers in reverse order
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output, learning_rate)
    
    def fit(self, X, y, epochs, learning_rate):
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss (mean squared error)
            loss = np.mean((output - y) ** 2)
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
            
            # Backward pass
            d_output = 2 * (output - y) / y.size
            self.backward(d_output, learning_rate)
    
    def predict(self, X):
        # Predict outputs for given inputs
        return self.forward(X)
