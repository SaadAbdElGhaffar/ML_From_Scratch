import numpy as np
from collections import Counter
from DecisionTree import DecisionTreeClassifier
# Bagging Class
class Bagging:
    def __init__(self, min_samples_split=2, max_depth=2, n_trees=20, sample_size_per_tree=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.sample_size_per_tree = sample_size_per_tree
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            X_sample, y_sample = self._bootstrap_sample(X, y, bag_size=self.sample_size_per_tree)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y, bag_size=100):
        tot_num_of_observations = X.shape[0]
        random_indices = np.random.choice(tot_num_of_observations, size=bag_size, replace=True)
        return X[random_indices], y[random_indices]

    def predict(self, X):
        tree_predictions = np.zeros((X.shape[0], self.n_trees))
        for i, tree in enumerate(self.trees):
            tree_predictions[:, i] = [tree.make_prediction(x, tree.root) for x in X]
        
        final_predictions = [self._most_common_label(tree_predictions[i]) for i in range(X.shape[0])]
        return final_predictions

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common