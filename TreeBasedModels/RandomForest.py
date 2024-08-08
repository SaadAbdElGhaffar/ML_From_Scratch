import numpy as np
from collections import Counter
from DecisionTree import DecisionTreeClassifier
from Bagging import Bagging

# Random Forest Classifier
class RandomForestClassifier(Bagging):
    def __init__(self, min_samples_split=2, max_depth=2, n_trees=20, sample_size_per_tree=100, max_features=None):
        super().__init__(min_samples_split=min_samples_split, max_depth=max_depth, n_trees=n_trees, sample_size_per_tree=sample_size_per_tree)
        self.max_features = max_features

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split, max_depth=self.max_depth, max_features=self.max_features)
            X_sample, y_sample = self._bootstrap_sample(X, y, bag_size=self.sample_size_per_tree)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)