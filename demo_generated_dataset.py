import matplotlib.pyplot as plt
from network.dense import Dense
from network.neural_network import NeuralNetwork
from network.relu import ReLU
from network.sigmoid import Sigmoid
import numpy as np
from network.utils import load_model, save_model


def generate_dataset(num_samples=1000):
    # Generate data for two clusters
    np.random.seed(42)
    cluster_1 = np.random.randn(num_samples//2, 2) + np.array([2, 2])
    cluster_2 = np.random.randn(num_samples//2, 2) + np.array([-2, -2])
    
    # Labels
    labels_1 = np.zeros((num_samples//2, 1))
    labels_2 = np.ones((num_samples//2, 1))
    
    # Combine data and labels
    X = np.vstack((cluster_1, cluster_2))
    y = np.vstack((labels_1, labels_2))
    
    # Shuffle the dataset
    shuffle_index = np.random.permutation(num_samples)
    X, y = X[shuffle_index], y[shuffle_index]
    
    return X, y

# Generate and plot the dataset
X, y = generate_dataset()
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='bwr')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Dataset')

# Create the neural network
nn = NeuralNetwork()
nn.add(Dense(2, 4, activation=ReLU()))
nn.add(Dense(4, 4, activation=ReLU()))
nn.add(Dense(4, 1, activation=Sigmoid()))

# Generate larger dataset
X, y = generate_dataset(num_samples=1000)

# Normalize data
X = X / np.max(np.abs(X), axis=0)

# Train the neural network
nn.fit(X, y, epochs=10000, learning_rate=0.01)

# Save the trained model
save_model(nn, 'binary_classification_model.pkl')

# Load the trained model
nn_loaded = load_model('binary_classification_model.pkl')

# Predict using the loaded model
predictions_loaded = nn_loaded.predict(X)
predictions_loaded = (predictions_loaded > 0.5).astype(int)  # Convert to binary output
accuracy = np.mean(predictions_loaded == y)
print(f"Accuracy of the loaded model: {accuracy * 100:.2f}%")

# Plot training loss
plt.figure()
plt.plot(nn.losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')

# plot the dataset with predictions
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=predictions_loaded.flatten(), cmap='bwr')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Dataset with Predictions')
plt.show()
