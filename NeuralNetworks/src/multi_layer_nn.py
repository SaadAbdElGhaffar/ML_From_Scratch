class MultiLayerNN:
    def __init__(self, weights_hidden, bias_hidden, weights_output, bias_output):
        self.weights_hidden = weights_hidden
        self.bias_hidden = bias_hidden
        self.weights_output = weights_output
        self.bias_output = bias_output

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_output) + self.bias_output
        self.output = sigmoid(self.output_input)
        return self.output

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        output_error = self.output - y
        output_delta = output_error * sigmoid_derivative(self.output_input)

        hidden_error = np.dot(output_delta, self.weights_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_input)

        self.weights_output -= learning_rate * np.dot(self.hidden_output.T, output_delta) / m
        self.bias_output -= learning_rate * np.sum(output_delta, axis=0, keepdims=True) / m
        self.weights_hidden -= learning_rate * np.dot(X.T, hidden_delta) / m
        self.bias_hidden -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True) / m

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

    def predict(self, X):
        return np.round(self.forward(X))