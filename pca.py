# This file shows an example of using PCA
from import_data import import_data
from sklearn.decomposition import PCA

(classes, data) = import_data()  # Import data from csv file
# print len(data[0])

# Specify the number of principal components to use (how many leading singular vector) to project on
pca = PCA(n_components=10)
X = pca.fit_transform(data) # Return the projected data
print len(classes)
print len(X)