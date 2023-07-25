import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import LocallyLinearEmbedding

iris = load_iris()
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=3)
lle.fit(iris.data)
data = lle.transform(iris.data)
target = iris.target
color = np.where(target == 1, 'r', 'm')
color = np.where(target == 2, 'g', color)
color = np.where(target == 0, 'b', color)
plt.scatter(data[:, 0], data[:, 1], color=color)
plt.show()