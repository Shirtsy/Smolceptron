class Perceptron:
    def __init__(self, size: int, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.weights = [0] * size
    
    def inference(self, input):
        # Multiple inputs by weights
        net_raw = 0
        for i, value in enumerate(input):
            net_raw += value * self.weights[i]
        
        #
        net = weighted_sum(net_raw)