import csv
from smolceptron import Perceptron

# Load training data
training_data = []
with open('data/iris_training.csv', mode = 'r')as file:
  csvFile = csv.reader(file)
  for line in csvFile:
        training_data.append({
            "input": [float(x) for x in line[:4]],
            "value": int(line[4]),
            "name": line[5]
        })
# Load validation data
validation_data = []
with open('data/iris_valid.csv', mode ='r')as file:
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
    print("--- Training Round", x+1, "---")
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