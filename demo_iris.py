from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from network.dense import Dense
from network.neural_network import NeuralNetwork
from network.relu import ReLU
import numpy as np
from network.softmax import Softmax
from network.utils import load_model, save_model
import matplotlib.pyplot as plt


# Load and preprocess the Iris dataset
def load_and_preprocess_iris():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)

    # One-hot encode the target labels
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Load and preprocess the Iris dataset
X_train, X_test, y_train, y_test = load_and_preprocess_iris()

# Create the neural network
nn = NeuralNetwork()
nn.add(Dense(4, 8, activation=ReLU()))  # First hidden layer
nn.add(Dense(8, 8, activation=ReLU()))  # Second hidden layer
nn.add(Dense(8, 3, activation=Softmax()))  # Output layer with softmax activation

# Train the neural network
nn.fit(X_train, y_train, epochs=5000, learning_rate=0.01)



# Save the trained model
save_model(nn, 'iris_classification_model.pkl')

# Load the trained model
nn_loaded = load_model('iris_classification_model.pkl')

# Predict using the loaded model
predictions = nn_loaded.predict(X_test)

predicted_classes = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

accuracy = np.mean(predicted_classes == true_labels)
print(f"Accuracy of the loaded model: {accuracy * 100:.2f}%")

# Plot training loss
plt.figure()
plt.plot(nn.losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
