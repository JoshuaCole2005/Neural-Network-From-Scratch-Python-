import numpy as np

def mse(true, prediction):
    return np.mean(np.power(true - prediction, 2))

def mse_deriv(true, prediction):
    return 2 * (prediction - true) / np.size(true)

def bce(true, prediction):
    return np.mean(-true * np.log(prediction) - (1 - true) * np.log(1 - prediction))

def bce_deriv(true, prediction):
    return ((1 - true) / (1 - prediction) - true / prediction) / np.size(true)
