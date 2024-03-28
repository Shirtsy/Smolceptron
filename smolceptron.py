import csv

class Perceptron:
    def __init__(self, size: int, learning_rate: float = 0.01, bias: float = 1.0):
        self.learning_rate = learning_rate
        self.bias = bias
        self.weights = [0.0] * size
    
    def inference(self, input: list):
        if len(input) != len(self.weights):
            raise ValueError("Input size must match model size")
        # Multiply inputs by weights and sum them
        net = 1 * self.bias
        for i, value in enumerate(input):
            net += value * self.weights[i]
        # Step function
        if net > 0:
            return 1
        else:
            return -1
        
    def update_weights(self, predicted_value: int, actual_value: int, input: list):
        if len(input) != len(self.weights):
            raise ValueError("Input size must match model size")
        for i, value in enumerate(input):
            self.weights[i] += self.learning_rate * (actual_value - predicted_value) * value