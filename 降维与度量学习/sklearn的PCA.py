import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
pca = PCA(n_components=2)
pca.fit(iris.data)
data = pca.transform(iris.data)
target = iris.target
color = np.where(target == 1, 'r', 'm')
color = np.where(target == 2, 'g', color)
color = np.where(target == 0, 'b', color)
plt.scatter(data[:, 0], data[:, 1], color=color)
plt.show()