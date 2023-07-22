import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


# 这里以均链接为例, 下面是没有优化过的代码
class AGNESModel:
    def __init__(self, k):
        self.k = k
        self.C = None

    def fit(self, X):
        X = np.array(X)
        C = []
        for xj in X:
            C.append(np.array([xj]))
        q = len(C)
        while q > self.k:
            # 找出最近的两个聚类簇
            centers = []
            for Ck in C:
                centers.append(np.mean(Ck, axis=0))
            min_i, min_j = AGNESModel.find_nearest(centers)
            for item in C[min_j]:
                C[min_i] = np.vstack((C[min_i], item))
            for j in range(min_j, q - 1):
                C[j] = C[j + 1]
            C.pop()
            q -= 1
        self.C = C

    @staticmethod
    def find_nearest(vectors):
        vectors = np.array(vectors)
        min_dist = -1
        min_i = -1
        min_j = -1
        length = vectors.shape[0]
        for i in range(length):
            for j in range(i + 1, length):
                dist = np.linalg.norm(vectors[i] - vectors[j])
                if min_dist < 0:
                    min_dist = dist
                    min_i = i
                    min_j = j
                elif dist < min_dist:
                    min_dist = dist
                    min_i = i
                    min_j = j
        return min_i, min_j


def main():
    iris = load_iris()
    data = iris.data

    model = AGNESModel(3)
    model.fit(data)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for k, Ck in enumerate(model.C):
        print(Ck)
        plt.scatter(Ck[:, 0], Ck[:, 1], color=colors[k])
    plt.show()


if __name__ == '__main__':
    main()
