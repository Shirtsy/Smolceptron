import csv

class Perceptron:
    def __init__(self, size: int, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.weights = [0] * size
    
    def inference(self, input: list):
        if len(input) != len(self.weights):
            raise ValueError("Input size must match model size")
        # Multiply inputs by weights and sum them
        net = 0
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


# Load training data
training_data = []
with open('iris_training.csv', mode = 'r')as file:
  csvFile = csv.reader(file)
  for line in csvFile:
        training_data.append({
            "input": [float(x) for x in line[:4]],
            "value": int(line[4]),
            "name": line[5]
        })
# Load validation data
validation_data = []
with open('iris_valid.csv', mode ='r')as file:
  csvFile = csv.reader(file)
  for line in csvFile:
        validation_data.append({
            "input": [float(x) for x in line[:4]],
            "value": int(line[4]),
            "name": line[5]
        })

perceptor = Perceptron(size = 4)

# Training
for x in range(3):
    correct = 0
    incorrect = 0
    for data in training_data:
        inference = perceptor.inference(data["input"])
        if inference == data["value"]:
            correct += 1
        else:
            incorrect += 1
        perceptor.update_weights(
            predicted_value = inference,
            actual_value = data["value"],
            input = data["input"]
        )
    print("---Training Round:", x, "---")
    print("Correct:", correct)
    print("Incorrect:", incorrect)

# Validation
correct = 0
incorrect = 0
for data in validation_data:
    inference = perceptor.inference(data["input"])
    if inference == data["value"]:
        correct += 1
    else:
        incorrect += 1
    perceptor.update_weights(
        predicted_value = inference,
        actual_value = data["value"],
        input = data["input"]
    )
print("-----Validation-----")
print("Correct:", correct)
print("Incorrect:", incorrect)

print("Weights:", [format(x, '.4f') for x in perceptor.weights])