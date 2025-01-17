1. Define the Structure of the MLP

   The MLP class constructor initializes the input size, hidden size, and output size. It also initializes the weights and biases with random values.

2. Implement Activation Functions

   The sigmoid function is used as the activation function, and its derivative is used for backpropagation.

3. Forward Propagation

   The forward method calculates the outputs of the hidden and output layers. It involves matrix multiplication of inputs and weights, adding biases, and applying the activation function.

4. Backward Propagation (Backpropagation)

   The backward method updates the weights and biases based on the error between the predicted and actual outputs. The error is propagated backward through the network, and gradients are calculated using the sigmoid derivative.

5. Training the Network

  The train method repeatedly performs forward and backward propagation for a specified number of epochs.

6. Predict Function
   
   The predict method uses the trained weights and biases to compute the output for new inputs.

7. Example Usage

  In the main block, we define the XOR problem, train the MLP on this data, and print the predictions.
"""

import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.random.randn(self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.random.randn(self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Input to Hidden Layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        # Hidden to Output Layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_input)

        return self.output

    def backward(self, X, y, output, learning_rate):
        # Calculate output layer error
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

    def predict(self, X):
        return self.forward(X)

# Example usage:
if __name__ == "__main__":
    # XOR Problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    mlp = MLP(input_size=2, hidden_size=2, output_size=1)
    mlp.train(X, y, epochs=10000, learning_rate=0.1)

    predictions = mlp.predict(X)
    print("Predictions:")
    print(predictions)
