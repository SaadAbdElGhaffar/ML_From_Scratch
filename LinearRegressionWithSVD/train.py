import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

ridge = RidgeCustom(alpha= 0.0001)
ridge.fit(X_train,y_train)

predictions = ridge.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f'MSE : {mse}')    


from sklearn.linear_model import Ridge
sklearn_ridge = Ridge(alpha = 0.0001, fit_intercept=False)
sklearn_ridge.fit(X_train, y_train)
y_ridge_sklearn = sklearn_ridge.predict(X_test)
y_ridge_sklearn

mse = mean_squared_error(y_test, y_ridge_sklearn)
print(f'MSE : {mse}')    