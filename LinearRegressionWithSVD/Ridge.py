import numpy as np
class RidgeCustom:
    def __init__(self , alpha = 0.01):
        self.alpha = alpha
        self.ridge_coeffs = None
    def fit(self, X, y):
        self.ridge_coeffs = (np.linalg.inv(X.T.dot(X) +
               (self.alpha * np.identity(X.shape[1])))).dot(X.T.dot(y))
    def predict(self, X):
        y_predicted = np.dot(X , self.ridge_coeffs).flatten()
        return y_predicted