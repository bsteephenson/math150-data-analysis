from utils import import_data
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA

(classes, data) = import_data()

TRAIN_SIZE = int(len(data) * .90) # Determine train size
indices = np.random.permutation(len(classes))
data = data[indices]
classes = classes[indices]

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

PCA_reduction = False

# Normal PCA
# pca = PCA(n_components=10)
# X = pca.fit(train_data)
# train_data_transformed = pca.transform(train_data)
# test_data_transformed = pca.transform(test_data)

# Kernel PCA
# kernelpca = KernelPCA(kernel="rbf", fit_in_verse_transform=True, gamma=10)
# X = kernelpca.fit(train_data)
# train_data_transformed = kernelpca.transform(train_data)
# test_data_transformed = kernelpca.transform(test_data)


# results = []

# for neighbor_num in range(1,25):
# 	for n in range(1,31):
# 		pca = PCA(n_components= n)
# 		pca.fit(train_data)
# 		train_data_transformed = pca.transform(train_data)
# 		test_data_transformed = pca.transform(test_data)

# 		# Perform classification
# 		classifier = kNN(n_neighbors=neighbor_num)
# 		classifier.fit(train_data_transformed, train_classes)

# 		# Perform prediction
# 		prediction = classifier.predict(test_data_transformed)
# 		correct = 0
# 		count = len(test_classes)
# 		for i, x in enumerate(test_classes):
# 		    if prediction[i] == x:
# 		        correct += 1
# 		results.append([int(neighbor_num),int(n),float(correct)/count])

# a = np.asarray(results)
# np.savetxt("./More_results_knn/knn10.csv", a, delimiter=",")

# optimal_n = 8
# optimal_k = 7

# pca = PCA(n_components= optimal_n)
# pca.fit(train_data)
# train_data_transformed = pca.transform(train_data)
# test_data_transformed = pca.transform(test_data)

# # Perform classification
# classifier = kNN(n_neighbors=optimal_k)
# classifier.fit(train_data_transformed, train_classes)

# # Perform prediction
# prediction = classifier.predict(test_data_transformed)
# correct = 0
# count = len(test_classes)
# for i, x in enumerate(test_classes):
#     if prediction[i] == x:
#         correct += 1
# print int(optimal_k),int(optimal_n),float(correct)/count
