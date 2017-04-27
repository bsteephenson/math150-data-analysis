
from utils import import_data
from sklearn.decomposition import PCA
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() 

(classes, data) = import_data()

TRAIN_SIZE = int(len(data) * .90) # Determine train size

indices = np.random.permutation(len(classes))
data = data[indices]
classes = classes[indices]

# PCA
# pca = PCA(n_components= n)
# 	pca.fit(train_data)
# 	train_data_transformed = pca.transform(train_data)
# 	test_data_transformed = pca.transform(test_data)

# Extract training and testing data
train_classes = classes[:TRAIN_SIZE]
train_data = data[:TRAIN_SIZE]
test_classes = classes[TRAIN_SIZE:]
test_data = data[TRAIN_SIZE:]

# Perform scaling
scaler.fit(train_data)
transformed_train_data = scaler.transform(train_data)
transformed_test_data = scaler.transform(test_data)

# num_layer = 5
# num_nodes = 10

# results = []

# for num_layer in range(1,7):
# 	for num_nodes in range(1,21):
# 		# Perform classification
# 		classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(num_layer, num_nodes), random_state=1)
# 		classifier.fit(transformed_train_data, train_classes)
# 		prediction = classifier.predict(transformed_test_data)

# 		# Measure accuracy on test set
# 		correct = 0
# 		count = len(test_classes)
# 		for i, x in enumerate(test_classes):
# 		    if prediction[i] == x:
# 		        correct += 1
# 		results.append([num_layer,num_nodes,float(correct)/count])


# a = np.asarray(results)
# np.savetxt("./More_results_nn/neural10.csv", a, delimiter=",")



opt_num_layer = 1
opt_num_nodes = 15

classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(opt_num_layer, opt_num_nodes), random_state=1)
classifier.fit(transformed_train_data, train_classes)
prediction = classifier.predict(transformed_test_data)

# Measure accuracy on test set
correct = 0
count = len(test_classes)
for i, x in enumerate(test_classes):
    if prediction[i] == x:
        correct += 1
print opt_num_layer,opt_num_nodes,float(correct)/count
