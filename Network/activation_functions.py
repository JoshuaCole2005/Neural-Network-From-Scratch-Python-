import numpy as np

class activation_function():
    def calc(self, x):
        raise NotImplementedError
    
    def calc_deriv(self, x):
        raise NotImplementedError

class relu(activation_function):
    def calc(self, x):
        return np.maximum(0.,x)
    
    def calc_deriv(self, x):
        return np.greater(x, 0.).astype(np.float32)

class sigmoid(activation_function):
    def calc(self, x):
        return 1/(1 + np.exp(-x))
    
    def calc_deriv(self, x):
        return self.calc(x) * (1 - self.calc(x))

class tanh(activation_function):
    def calc(self, x):
        return np.tanh(x)
    
    def calc_deriv(self, x):
        return 1-np.tanh(x)**2
