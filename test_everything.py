import utils

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neural_network import MLPClassifier


classes, data = utils.import_data()

# Each element of the results table looks like
# [trial, k, train_percent, dimensions, test_accuracy]

results_table = []

for trial in range(0, 5):
	print "trial:", trial

	indices = np.random.permutation(len(classes))
	data = data[indices]
	classes = classes[indices]

	samples = len(data)

	for k in range(1, 8):
		# print "k = ", k
		for train_percent in [10, 25, 50, 60, 70, 80, 90, 95]:
			
			# print "train_percent = ", train_percent
			
			TRAIN_SIZE = int(samples * train_percent / 100)

			# Extract training and test files
			train_classes = classes[:TRAIN_SIZE]
			train_data = data[:TRAIN_SIZE]
			test_classes = classes[TRAIN_SIZE:]
			test_data = data[TRAIN_SIZE:]
			
			# Perform scaling
			scaler = StandardScaler() 
			scaler.fit(train_data)
			train_data = scaler.transform(train_data)
			test_data = scaler.transform(test_data)

			for dimensions in [1, 2, 4, 8, 16, 30]:
				# print "dimensions = ", dimensions

				# reduce dimensions of test and training data
				pca = PCA(n_components = dimensions)
				pca.fit(train_data)

				low_dim_train_data = pca.transform(train_data)
				low_dim_test_data = pca.transform(test_data)

				# Perform classification
				classifier = kNN(n_neighbors=k)
				classifier.fit(low_dim_train_data, train_classes)

				test_accuracy = classifier.score(low_dim_test_data, test_classes)

				results_table.append([trial, k, train_percent, dimensions, test_accuracy])


a = np.asarray(results_table)
np.savetxt("knn_results.csv", a, delimiter=",")


print "doing SVM now"


# Each element of the results table looks like
# [trial, train_percent, dimensions, test_accuracy]

results_table = []

for trial in range(0, 5):
	print "trial:", trial

	indices = np.random.permutation(len(classes))
	data = data[indices]
	classes = classes[indices]

	samples = len(data)

	for train_percent in [10, 25, 50, 60, 70, 80, 90, 95]:
		
		# print "train_percent = ", train_percent
		
		TRAIN_SIZE = int(samples * train_percent / 100)

		# Extract training and test files
		train_classes = classes[:TRAIN_SIZE]
		train_data = data[:TRAIN_SIZE]
		test_classes = classes[TRAIN_SIZE:]
		test_data = data[TRAIN_SIZE:]
		
		# Perform scaling
		scaler = StandardScaler() 
		scaler.fit(train_data)
		train_data = scaler.transform(train_data)
		test_data = scaler.transform(test_data)

		for dimensions in [1, 2, 4, 8, 16, 30]:
			# print "dimensions = ", dimensions

			# reduce dimensions of test and training data
			pca = PCA(n_components = dimensions)
			pca.fit(train_data)

			low_dim_train_data = pca.transform(train_data)
			low_dim_test_data = pca.transform(test_data)

			# Perform classification
			classifier = svm.LinearSVC()
			classifier.fit(low_dim_train_data, train_classes)

			test_accuracy = classifier.score(low_dim_test_data, test_classes)

			results_table.append([trial, train_percent, dimensions, test_accuracy])


a = np.asarray(results_table)
np.savetxt("svm_results.csv", a, delimiter=",")



print "doing neural now"


# Each element of the results table looks like
# [trial, train_percent, dimensions, hidden_units, test_accuracy]

results_table = []

for trial in range(0, 5):
	print "trial:", trial
	indices = np.random.permutation(len(classes))
	data = data[indices]
	classes = classes[indices]

	samples = len(data)

	for train_percent in [10, 25, 50, 60, 70, 80, 90, 95]:
		
		print "train_percent = ", train_percent
		
		TRAIN_SIZE = int(samples * train_percent / 100)

		# Extract training and test files
		train_classes = classes[:TRAIN_SIZE]
		train_data = data[:TRAIN_SIZE]
		test_classes = classes[TRAIN_SIZE:]
		test_data = data[TRAIN_SIZE:]
		
		# Perform scaling
		scaler = StandardScaler() 
		scaler.fit(train_data)
		train_data = scaler.transform(train_data)
		test_data = scaler.transform(test_data)

		for dimensions in [1, 2, 4, 8, 16, 30]:
			print "dimensions = ", dimensions

			# reduce dimensions of test and training data
			pca = PCA(n_components = dimensions)
			pca.fit(train_data)

			low_dim_train_data = pca.transform(train_data)
			low_dim_test_data = pca.transform(test_data)

			for hidden_units in [1, 2, 4, 8]:
				# Perform classification
				classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(hidden_units), random_state=1)
				classifier.fit(low_dim_train_data, train_classes)

				test_accuracy = classifier.score(low_dim_test_data, test_classes)

				results_table.append([trial, train_percent, dimensions, hidden_units, test_accuracy])


a = np.asarray(results_table)
np.savetxt("nueral_net_results.csv", a, delimiter=",")


