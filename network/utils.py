import pickle

# Function to save the trained model to a file
def save_model(nn, filename):
    with open(filename, 'wb') as file:
        pickle.dump(nn, file)

# Function to load a trained model from a file
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
