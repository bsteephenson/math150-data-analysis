from import_data import import_data
import numpy as np

(classes, data) = import_data()

# randomly split the data into training data and test data

TRAIN_SIZE = int(len(data) * .95)

indices = np.random.permutation(len(classes))
data = data[indices]
classes = classes[indices]

train_classes = classes[:TRAIN_SIZE]
train_data = data[:TRAIN_SIZE]

test_classes = classes[TRAIN_SIZE:]
test_data = data[TRAIN_SIZE:]

# Make the svm classifier

from sklearn import svm
classifier = svm.LinearSVC()
classifier.fit(train_data, train_classes)
print "predicted", classifier.predict(test_data)
print "expected", test_classes
print "training data score:", classifier.score(train_data, train_classes)
print "test data score:", classifier.score(test_data, test_classes)

