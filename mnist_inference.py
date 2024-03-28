import csv
from smolceptron import Perceptron
from PIL import Image

def get_pixel_values_flat(image_path):
    # Open the image
    with Image.open(image_path) as img:
        # Convert the image to grayscale ('L' mode for 8-bit pixels, black and white)
        img = img.convert('L')
        # Retrieve all pixel values as a flat list
        pixels = list(img.getdata())
        return pixels

def is_equal(in_1, in_2) -> int:
    """Returns 1 for equality. -1 for inequality."""
    if in_1 == in_2:
        return 1
    else:
        return -1

# Load weight data
all_weights = []
with open('mnist_test_weights.csv', mode = 'r')as file:
  csvFile = csv.reader(file)
  for line in csvFile:
        all_weights.append([float(x) for x in line])

perceptors = []
for weights in all_weights:
    perceptor = Perceptron(size = len(weights))
    perceptor.weights = weights
    perceptors.append(perceptor)

pixels = get_pixel_values_flat("image.png")
#print(pixels)
for i, perceptor in enumerate(perceptors):
    inference = perceptor.inference(pixels)
    print(str(i) + ":" + " " + str(inference))