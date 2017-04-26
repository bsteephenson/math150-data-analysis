# This file shows an example of using PCA
from import_data import import_data
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA


(classes, data) = import_data()  # Import data from csv file
# print len(data[0])

# Specify the number of principal components to use (how many leading singular vector) to project on
# This is the usual PCA
pca = PCA(n_components=10)
X = pca.fit_transform(data) # Return the projected data
print len(classes)
print len(X[0])


def normal_pca(n_components=5):
	(classes, data) = import_data()  # Import data from csv file
	pca = PCA(n_components=n)
	X = pca.fit_transform(data) # Return the projected data
	return (classes, X)


def kernel_pca():
	(classes, data) = import_data()
	kernelpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
	X = kernelpca.fit_transform(data)
	return (classes, X)
