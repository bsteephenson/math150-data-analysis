# This is to test out an idea for classification

# When we're classifying a point p, this creates a box around p and looks at
# the training data classes within that box.
# We use the standard deviation of each dimension to create the box

from import_data import import_data
import numpy as np

(classes, data) = import_data()

# randomly split the data into training data and test data

TRAIN_SIZE = int(len(data) * .95)

np.random.seed()
indices = np.random.permutation(len(classes))
print indices

# exit()
data = data[indices]
classes = classes[indices]

train_classes = classes[:TRAIN_SIZE]
train_data = data[:TRAIN_SIZE]

test_classes = classes[TRAIN_SIZE:]
test_data = data[TRAIN_SIZE:]

# Find the standard deviation of each dimension

stddevs = np.std(train_data, axis = 0) * 2

def classify(point):
	count = 0.0
	positives = 0.0
	for i in range(TRAIN_SIZE):
		if (np.abs(point - train_data[i]) < stddevs).all():
			count = count + 1
			if (train_classes[i] > 0):
				positives = positives + 1
	if (count == 0):
		return 0
	else:
		return positives / count

hits = 0.0
misses = 0.0
border = 0.0

for i in range(len(test_classes)):
	predicted_class = classify(test_data[i])
	print "expected:", predicted_class, " actual class:", test_classes[i]
	if (predicted_class == .5):
		border = border + 1
	elif (predicted_class > .5 and test_classes[i] > 0):
		hits = hits + 1
	elif (predicted_class < .5 and test_classes[i] < 0):
		hits = hits + 1
	else:
		misses = misses + 1

print "hits", hits, "misses", misses, "border", border

print "accuracy:", (hits / (hits + misses + border))
