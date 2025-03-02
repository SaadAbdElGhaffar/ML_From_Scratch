class SingleLayerNN:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, X):
        self.output_input = np.dot(X, self.weights) + self.bias
        self.output = sigmoid(self.output_input)
        return self.output

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        output_error = self.output - y
        output_delta = output_error * sigmoid_derivative(self.output_input)
        self.weights -= learning_rate * np.dot(X.T, output_delta) / m
        self.bias -= learning_rate * np.sum(output_delta, axis=0, keepdims=True) / m

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

    def predict(self, X):
        return np.round(self.forward(X))