from network.dense import Dense
from network.neural_network import NeuralNetwork
from network.relu import ReLU
from network.sigmoid import Sigmoid
import numpy as np
from network.utils import load_model, save_model


# Example usage:
if __name__ == "__main__":
    # Create the neural network
    nn = NeuralNetwork()
    nn.add(Dense(2, 4, activation=ReLU()))
    nn.add(Dense(4, 4, activation=ReLU()))
    nn.add(Dense(4, 1, activation=Sigmoid()))
    
    # Sample data (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Normalize data
    X = X / np.max(X)
    y = y / np.max(y)
    
    # Train the neural network
    nn.fit(X, y, epochs=100000, learning_rate=0.05)

    # Save the trained model
    save_model(nn, 'xor_model.pkl')

    # Load the trained model
    nn_loaded = load_model('xor_model.pkl')

    # Predict
    predictions = nn_loaded.predict(X)
    print("Predictions:")
    print(predictions)
