
# Neural Network Demo with numpy


This project implements a simple neural network using `numpy`. The neural network is designed in an object-oriented style and includes functionalities for saving and loading the model.

## Getting Started

### Prerequisites

- Python 3.x
- `numpy` library
- `matplotlib` library (for visualizing the dataset)

You can install requirments using pip:
```bash
pip install -r requirements.txt
```

## Example Usage

Here is an example of how to use the Neural Network:

```python
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

# Predict using the loaded model
predictions_loaded = nn_loaded.predict(X)
print("Predictions from loaded model:")
print(predictions_loaded)
```

