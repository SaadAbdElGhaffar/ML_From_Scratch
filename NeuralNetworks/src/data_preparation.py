import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def generate_dataset(n_samples=1000, noise=0.2, random_state=42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y

def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)