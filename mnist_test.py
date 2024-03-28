import csv
from smolceptron import Perceptron

def is_equal(in_1, in_2) -> int:
    """Returns 1 for equality. -1 for inequality."""
    if in_1 == in_2:
        return 1
    else:
        return -1

def greater_than(in_1, in_2) -> int:
    """Returns 1 for greater. Else 0."""
    if in_1 > in_2:
        return 1
    else:
        return 0

# Load training data
training_data = []
with open('data/mnist_train.csv', mode = 'r')as file:
  next(file)
  csvFile = csv.reader(file)
  for line in csvFile:
        training_data.append({
            "input": [int(x) for x in line[1:]],
            "value": is_equal(int(line[0]), 0)
        })
# Load validation data
validation_data = []
with open('data/mnist_test.csv', mode ='r')as file:
  next(file)
  csvFile = csv.reader(file)
  for line in csvFile:
        validation_data.append({
            "input": [int(x) for x in line[1:]],
            "value": is_equal(int(line[0]), 0)
        })

perceptor = Perceptron(size = len(training_data[0]["input"]),
                       learning_rate = 0.01)

# Training
for x in range(5):
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
    print("--- Training Round", x+1, "---")
    print("Correct:", correct)
    print("Incorrect:", incorrect)
    print("Accuracy:", format(correct/(correct + incorrect), '.6f'))
    print("Learning Rate:", perceptor.learning_rate)

# Validation
correct = 0
incorrect = 0
for data in validation_data:
    inference = perceptor.inference(data["input"])
    if inference == data["value"]:
        correct += 1
    else:
        incorrect += 1
print("-----Validation-----")
print("Correct:", correct)
print("Incorrect:", incorrect)
print("Accuracy:", format(correct/(correct + incorrect), '.6f'))

#print("Weights:", [format(x, '.2f') for x in perceptor.weights])