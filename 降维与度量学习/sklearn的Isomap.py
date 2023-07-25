import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import Isomap

'''
Isomap常用于可视化
如果要对新样本进行映射, 需要训练一个回归学习器
'''
iris = load_iris()
iso = Isomap(n_components=2, n_neighbors=3)
iso.fit(iris.data)
data = iso.transform(iris.data)
target = iris.target
color = np.where(target == 1, 'r', 'm')
color = np.where(target == 2, 'g', color)
color = np.where(target == 0, 'b', color)
plt.scatter(data[:, 0], data[:, 1], color=color)
plt.show()