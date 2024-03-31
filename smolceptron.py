import csv
import math

class Perceptron:
    def __init__(self, size: int, learning_rate: float = 0.01, bias: float = 1.0):
        self.learning_rate = learning_rate
        self.bias = bias
        self.weights = [0.0] * size

    def sigmoid_ish(self, input: float) -> float:
        if input > 30:
            return 1
        elif input < -30:
            return -1
        else:
            return 2 * (1 / (1 + math.exp(-input))) - 1
    
    def inference(self, input: list):
        if len(input) != len(self.weights):
            raise ValueError("Input size must match model size")
        # Multiply inputs by weights and sum them
        net = 1 * self.bias
        for i, value in enumerate(input):
            net += value * self.weights[i]
        # Step function
        # if net > 0:
        #     return 1
        # else:
        #     return -1
        return self.sigmoid_ish(net)
        
    def update_weights(self, predicted_value: float, actual_value: float, input: list):
        if len(input) != len(self.weights):
            raise ValueError("Input size must match model size")
        for i, value in enumerate(input):
            self.weights[i] += self.learning_rate * (actual_value - predicted_value) * value