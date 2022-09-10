import numpy as np

class Fully_Connected_Layer():
    def __init__(self, input_size, output_size, activation):
        self.input = np.zeros(input_size, dtype = np.float32)
        self.activation_input = np.zeros(output_size, dtype = np.float32)
        self.output = np.zeros(output_size, dtype = np.float32)
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)
        self.activation = activation

    def forward_propogation(self, input):
        self.input = input
        y = (np.dot(self.weights, self.input)) + self.biases
        self.activation_input = y
        a = self.activation.calc(y)
        self.output = a
        return self.output
    
    def back_propogation(self, y_grad, lr):
        activation_grad = np.multiply(y_grad, self.activation.calc_deriv(self.activation_input))
        w_grad = np.dot(activation_grad, self.input.T)
        x_grad = np.dot(self.weights.T, activation_grad)
        self.weights -= lr * w_grad
        self.biases -= lr * activation_grad
        return x_grad


