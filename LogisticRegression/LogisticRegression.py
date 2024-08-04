import numpy as np

class LogisticRegressionCustom:
    def __init__(self, lr=0.01, max_iterations=1000, bias=True, threshold = 0.5):
      self.lr = lr
      self.max_iterations = max_iterations
      self.bias = bias
      self.threshold = threshold
      self.weights = None

    
    def initialize(self, X):
        if self.bias:
            self.weights = np.random.randn((X.shape[1] + 1, 1))
            X= np.c_[np.ones((X.shape[0], 1)), X]
        else:
            self.weights = np.random.randn((X.shape[1], 1))

        return self.weights, X  

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.weights, X = initialize(X, self.bias)
        for i in range(self.max_iterations):
            self.weights = self.weights - lr * X.T.dot(sigmoid(np.dot(X, self.weights)) - np.reshape(y, (len(y), 1)))
           
    def predict(self, X):
        z = np.dot(initialize(X, self.bias)[1], self.weights)
        preds = []
        for predicted_proba in sigmoid(z):
            if predicted_proba > self.threshold:
              preds.append(1)
            else:
              preds.append(0)
        return preds
      