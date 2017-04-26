
from import_data import import_data
from sklearn.decomposition import PCA
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() 

(classes, data) = import_data()

TRAIN_SIZE = int(len(data) * .80) # Determine train size

indices = np.random.permutation(len(classes))
data = data[indices]
classes = classes[indices]

# Extract training and testing data
train_classes = classes[:TRAIN_SIZE]
train_data = data[:TRAIN_SIZE]
test_classes = classes[TRAIN_SIZE:]
test_data = data[TRAIN_SIZE:]

# Perform scaling
scaler.fit(train_data)
transformed_train_data = scaler.transform(train_data)
transformed_test_data = scaler.transform(test_data)

for n in range(5,31):
	pca = PCA(n_components= n)
	pca.fit(train_data)
	train_data_transformed = pca.transform(train_data)
	test_data_transformed = pca.transform(test_data)

	# Perform classification
	classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 7), random_state=1)
	classifier.fit(transformed_train_data, train_classes)
	prediction = classifier.predict(transformed_test_data)

	# Measure accuracy on test set
	correct = 0
	count = len(test_classes)
	for i, x in enumerate(test_classes):
	    if prediction[i] == x:
	        correct += 1
	print "Number of PCA components=",n, "Accuracy:",float(correct)/count