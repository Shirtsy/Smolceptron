import csv
import random
from smolceptron import Perceptron

TRAINING_ROUNDS = 10
BASE_LEARNING_RATE = 0.01
BIAS = 1
TARGET = 1

def is_target(input) -> float:
    """Returns 1 for equality. -1 for inequality."""
    if input == TARGET:
        return 1
    elif input == 0:
        return -1
    else:
        return 0

def load_data(filepath: str, target: int) -> list[dict]:
    data = []
    with open(filepath, mode = 'r')as file:
        next(file)
        csvFile = csv.reader(file)
        for line in csvFile:
                data.append({
                    "input": [int(x) for x in line[1:]],
                    "value": is_target(int(line[0]))
                })
    # Cut down list of non-target values to make ratio 50:50
    targets = [x for x in data if x["value"] == 1]
    non_targets = [x for x in data if x["value"] == -1]
    random.shuffle(non_targets)
    data = targets + non_targets
    random.shuffle(data)
    return data

def train_mnist_perceptor(target: int, rounds: int, learning_rate: float) -> list[float]:
    training_data = load_data("data/mnist_train.csv", target)
    validation_data = load_data("data/mnist_test.csv", target)

    perceptor = Perceptron(size = len(training_data[0]["input"]), learning_rate=learning_rate, bias = BIAS)
    # Training
    for x in range(rounds):
        correct = 0
        incorrect = 0
        for data in training_data:
            inference = perceptor.inference(data["input"])
            # Detect if correct within threshold
            if inference - data["value"] < 0.1:
                correct += 1
            else:
                incorrect += 1
            perceptor.update_weights(
                predicted_value = inference,
                actual_value = data["value"],
                input = data["input"]
            )

            perceptor.learning_rate = BASE_LEARNING_RATE / (x+1)
        print("- Training Round:", x+1)
        print("\tCorrect:", correct)
        print("\tIncorrect:", incorrect)
        print("\tAccuracy:", format(correct/(correct + incorrect), '.6f'))
        print("\tLearning Rate:", format(perceptor.learning_rate, '.6f'))

    # Validation
    correct = 0
    incorrect = 0
    for data in validation_data:
        inference = perceptor.inference(data["input"])
        if inference == data["value"]:
            correct += 1
        else:
            incorrect += 1
    print("- Validation")
    print("\tCorrect:", correct)
    print("\tIncorrect:", incorrect)
    print("\tAccuracy:", format(correct/(correct + incorrect), '.6f'))
    print("")
    
    return perceptor.weights

all_weights = []
print("### Training for", TARGET, "###")
all_weights.append(train_mnist_perceptor(TARGET, TRAINING_ROUNDS, BASE_LEARNING_RATE))

with open("weights_0-1.csv", 'w', newline="") as file:
    csvFile = csv.writer(file)
    for weights in all_weights:
        csvFile.writerow(weights)
#print("Weights:", [format(x, '.2f') for x in perceptor.weights])