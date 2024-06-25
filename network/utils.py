import pickle

def save_model(nn, filename):
    with open(filename, 'wb') as file:
        pickle.dump(nn, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
