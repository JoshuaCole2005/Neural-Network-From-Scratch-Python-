import numpy as np

class feed_forward():
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
    
    def calc_loss(self, loss, true, prediction):
        return loss.calc(true, prediction), loss.calc_deriv(true, prediction)

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward_propogation(output)
        return output

    def train(self, training_input, training_output, epochs, lr, loss, loss_deriv):
        num_inputs = len(training_input)
        for epoch in range(epochs):
            error = 0
            for inputs, true_vals in zip(training_input, training_output):
                #Doing Forward Propogation
                predictions = self.predict(inputs)

                #calculating error
                error = error + loss(true_vals, predictions)

                #Doing Backward Propogation
                output_grad = loss_deriv(true_vals, predictions)
                for layer in reversed(self.layers):
                    output_grad = layer.back_propogation(output_grad, lr)
            
            error /= num_inputs
            print(f"Epoch {epoch + 1}/{epochs} \t Error: {error}")
