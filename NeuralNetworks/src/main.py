from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from single_layer_nn import SingleLayerNN
from multi_layer_nn import MultiLayerNN

# Generate dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Set the same random seed
seed = 42
np.random.seed(seed)

# Single Layer Comparison
initial_weights_single = np.random.randn(2, 1) * 0.01
initial_bias_single = np.zeros((1, 1))

single_layer_manual = SingleLayerNN(initial_weights_single.copy(), initial_bias_single.copy())
single_layer_manual.train(X_train, y_train, epochs=10000, learning_rate=0.1)
manual_single_predictions = single_layer_manual.predict(X_test)
manual_single_accuracy = accuracy_score(y_test, manual_single_predictions)

# Multi-Layer Comparison
initial_weights_hidden = np.random.randn(2, 4) * 0.01
initial_bias_hidden = np.zeros((1, 4))
initial_weights_output = np.random.randn(4, 1) * 0.01
initial_bias_output = np.zeros((1, 1))

multi_layer_manual = MultiLayerNN(initial_weights_hidden.copy(), initial_bias_hidden.copy(), initial_weights_output.copy(), initial_bias_output.copy())
multi_layer_manual.train(X_train, y_train, epochs=10000, learning_rate=0.1)
manual_multi_predictions = multi_layer_manual.predict(X_test)
manual_multi_accuracy = accuracy_score(y_test, manual_multi_predictions)

print(f"Manual Single Layer Accuracy: {manual_single_accuracy}")
print(f"Manual Multi Layer Accuracy: {manual_multi_accuracy}")