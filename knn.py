from import_data import import_data
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.decomposition import PCA

(classes, data) = import_data()

TRAIN_SIZE = int(len(data) * .90) # Determine train size

indices = np.random.permutation(len(classes))
data = data[indices]
classes = classes[indices]

train_classes = classes[:TRAIN_SIZE]
train_data = data[:TRAIN_SIZE]

test_classes = classes[TRAIN_SIZE:]
test_data = data[TRAIN_SIZE:]


dimension_redux = 30
# Perform PCA to reduce to specified dimension
if dimension_redux == 30:
	train_data_transformed = train_data
	test_data_transformed = test_data
else:
	pca = PCA(n_components=dimension_redux)
	train_data_transformed = pca.fit_transform(train_data)
	test_data_transformed = pca.fit_transform(test_data)


# Number of neighbors

for neighbor_num in range(1,10):
	# Train classifier
	classifier = kNN(n_neighbors=neighbor_num)
	classifier.fit(train_data_transformed, train_classes)

	# Perform prediction
	prediction = classifier.predict(test_data_transformed)
	correct = 0
	count = len(test_classes)
	for i, x in enumerate(test_classes):
	    if prediction[i] == x:
	        correct += 1
	        
	print "k=",neighbor_num, " ",float(correct)/count