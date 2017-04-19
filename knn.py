

from import_data import import_data
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNN

(classes, data) = import_data()

TRAIN_SIZE = int(len(data) * .95) # Determine train size

indices = np.random.permutation(len(classes))
data = data[indices]
classes = classes[indices]

train_classes = classes[:TRAIN_SIZE]
train_data = data[:TRAIN_SIZE]

test_classes = classes[TRAIN_SIZE:]
test_data = data[TRAIN_SIZE:]


neighbor_num = 10

classifier = kNN(n_neighbors=neighbor_num)
classifier.fit(train_data, train_classes)


prediction = classifier.predict(test_data)
correct = 0
count = len(test_classes)
for i, x in enumerate(test_classes):
    if prediction[i] == x:
        correct += 1
        
print float(correct)/count