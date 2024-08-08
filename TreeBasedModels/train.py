from sklearn import datasets
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTreeClassifier
import numpy as np

# Example usage
data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(min_samples_split=20, max_depth=100)
clf.fit(X=X_train, Y=Y_train)

prediction = clf.predict(X_train)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(Y_train, prediction)
print("Decision Tree Accuracy:", acc)

# Bagging example
bagging_clf = Bagging(min_samples_split=20, max_depth=100, n_trees=20, sample_size_per_tree=100)
bagging_clf.fit(X_train, Y_train)

bagging_prediction = bagging_clf.predict(X_train)
bagging_acc = accuracy(Y_train, bagging_prediction)
print("Bagging Accuracy:", bagging_acc)

# Random Forest
random_forest_clf = RandomForestClassifier(min_samples_split=20, max_depth=100, n_trees=20, sample_size_per_tree=100, max_features=int(np.sqrt(X_train.shape[1])))
random_forest_clf.fit(X_train, Y_train)

random_forest_prediction = random_forest_clf.predict(X_train)
random_forest_acc = accuracy(Y_train, random_forest_prediction)
print("Random Forest Accuracy:", random_forest_acc)
