import csv
from smolceptron import Perceptron

TRAINING_ROUNDS = 5
BASE_LEARNING_RATE = 0.5
BIAS = 1

def is_equal(in_1, in_2) -> float:
    """Returns 1 for equality. -1 for inequality."""
    if in_1 == in_2:
        return 1
    else:
        return -1

def train_mnist_perceptor(target: int, rounds: int, learning_rate: float) -> list[float]:
    # Load training data
    training_data = []
    with open('data/mnist_train.csv', mode = 'r')as file:
        next(file)
        csvFile = csv.reader(file)
        for line in csvFile:
                training_data.append({
                    "input": [int(x) for x in line[1:]],
                    "value": is_equal(int(line[0]), target)
                })
    # Load validation data
    validation_data = []
    with open('data/mnist_test.csv', mode ='r')as file:
        next(file)
        csvFile = csv.reader(file)
        for line in csvFile:
                validation_data.append({
                    "input": [int(x) for x in line[1:]],
                    "value": is_equal(int(line[0]), target)
                })

    perceptor = Perceptron(size = len(training_data[0]["input"]), learning_rate=learning_rate, bias = BIAS)
    initial_learning_rate = perceptor.learning_rate
    # Training
    for x in range(rounds):
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
        # Decrease learning rate over training rounds for fine-tuning
        perceptor.learning_rate = initial_learning_rate / (x + 1)

        print("- Training Round:", x+1)
        print("\tCorrect:", correct)
        print("\tIncorrect:", incorrect)
        print("\tAccuracy:", format(((correct/(correct + incorrect))*10)-9, '.6f'))
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
    print("\tAccuracy:", format(((correct/(correct + incorrect))*10)-9, '.6f'))
    print("")
    
    return perceptor.weights

all_weights = []
for i in range(10):
    print("### Training for", i, "###")
    all_weights.append(train_mnist_perceptor(i, TRAINING_ROUNDS, BASE_LEARNING_RATE))

with open('weights_0.csv', 'w', newline='') as file:
    csvFile = csv.writer(file)
    for weights in all_weights:
        csvFile.writerow(weights)
#print("Weights:", [format(x, '.2f') for x in perceptor.weights])