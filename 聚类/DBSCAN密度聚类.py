import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


class DBSCANModel:
    def __init__(self, eps, MinPts):
        self.eps = eps
        self.MinPts = MinPts
        self.k = 0
        self.C = []

    def fit(self, X):
        X = np.array(X)
        coreObjs = []  # 核心对象集合
        m = X.shape[0]

        # 找到所有核心对象
        for j in range(m):
            if self.getNeighbors(X, X[j], j).shape[0] >= self.MinPts:
                coreObjs.append(X[j])

        gamma = X  # 未访问样本集合 Γ = D
        while len(coreObjs) > 0:
            gamma_old = gamma.copy()
            o = DBSCANModel.select_random_vectors(coreObjs, 1)[0]
            Q = [o]
            gamma = DBSCANModel.diff_set(gamma, Q)
            while len(Q) > 0:
                q = Q.pop()
                neighbors = self.getNeighbors(X, q)
                if neighbors.shape[0] >= self.MinPts:
                    delta = DBSCANModel.inter_set(neighbors, gamma)
                    if len(delta) > 0:
                        Q.append(delta)
                        gamma = DBSCANModel.diff_set(gamma, delta)
            self.k += 1
            Ck = DBSCANModel.diff_set(gamma_old, gamma)
            self.C.append(np.array(Ck))
            # self.C.append(np.array(Ck))
            coreObjs = DBSCANModel.diff_set(coreObjs, Ck)

    def getNeighbors(self, X, xj, j=-1):
        if j < 0:
            for index, xi in enumerate(X):
                if np.allclose(xj, xi):
                    j = index
                    break
        neighbors = []
        for index, xi in enumerate(X):
            if index == j:
                continue
            dist = np.linalg.norm(X[j] - xi)
            if dist <= self.eps:
                neighbors.append(xi)
        return np.array(neighbors)

    # 判断是否包含向量
    @staticmethod
    def contain_vector(arr, v):
        for index, item in enumerate(arr):
            if np.allclose(v, item):
                return index
        return -1

    # 选择随机向量
    @staticmethod
    def select_random_vectors(arr, k):
        arr = np.array(arr)
        # 获取数组的形状
        num_vectors, vector_dim = arr.shape

        # 生成k个随机索引
        random_indices = np.random.choice(num_vectors, k, replace=False)

        # 根据随机索引选取向量
        random_vectors = arr[random_indices]

        return random_vectors

    # 计算集合间的差集
    @staticmethod
    def diff_set(X, Y):
        result = []
        for x in X:
            add = True
            for y in Y:
                if np.allclose(x, y):
                    add = False
                    break
            if add:
                result.append(x)

        return result

    # 计算交集
    @staticmethod
    def inter_set(X, Y):
        result = []
        for x in X:
            for y in Y:
                if np.allclose(x, y):
                    result.append(x)
                    break
        return result


def main():
    iris = load_iris()
    data = iris.data

    model = DBSCANModel(1.6, 4)
    model.fit(data)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for k, Ck in enumerate(model.C):
        plt.scatter(Ck[:, 0], Ck[:, 1], color=colors[k])
    plt.show()


if __name__ == '__main__':
    main()
