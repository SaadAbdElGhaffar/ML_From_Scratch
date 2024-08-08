from sklearn import datasets
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTreeClassifier
import numpy as np

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = DecisionTreeClassifier(min_samples_split=3, max_depth=7)
clf.fit(X=X_train, Y=y_train)

prediction = clf.predict(X_test)  # Fixed variable name

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, prediction)
print("Accuracy:", acc)
