
from import_data import import_data
from sklearn.decomposition import PCA

(classes, data) = import_data()  # Import data from csv file
# print len(data[0])

pca = PCA(n_components=10)
X = pca.fit_transform(data)