import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score


class LVQModel:
    def __init__(self):
        self.P = None
        self.T = None

    def fit(self, X, Y, q, lr=0.01, max_epoch=10000):
        P = None
        T = None
        # 初始化原型向量
        # 随机选取q个样本作为原型向量
        # q个样本的类别标记似要包含所有多分类任务中的所有类别标记
        while True:
            P, T = LVQModel.select_random_vectors_with_flag(X, Y, q)
            if len(np.unique(Y) == len(np.unique(T))):
                break

        num_vectors, _ = X.shape
        # 如果连续5轮更新比较小则跳出循环, 这里懒得跳出了
        for _ in range(max_epoch):
            # 随机选取样本
            j = np.random.choice(num_vectors, 1, replace=False)[0]

            # 获取最近的原型向量及距离
            i, distance = LVQModel.getNearestP(X[j], P)

            # 对原型向量进行更新
            if Y[j] == T[i]:
                flag = 1
            else:
                flag = -1

            P[i] = P[i] + flag * lr * (X[i] - P[i])

        self.P = P
        self.T = T

    @staticmethod
    def getNearestP(Xi, P):
        distances = np.linalg.norm(Xi - P, axis=1)
        shortest_index = np.argmin(distances)
        return shortest_index, distances[shortest_index]

    @staticmethod
    def select_random_vectors_with_flag(X, Y, q):
        X = np.array(X)
        Y = np.array(Y)
        # 获取数组的形状
        num_vectors, vector_dim = X.shape

        # 生成k个随机索引
        random_indices = np.random.choice(num_vectors, q, replace=False)

        # 根据随机索引选取向量
        random_vectors_X = X[random_indices]
        random_vectors_Y = Y[random_indices]

        return random_vectors_X, random_vectors_Y

    def predict(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            return self.__predict(X)
        else:
            result = []
            for i in range(X.shape[0]):
                result.append(self.__predict(X[i]))
            return np.array(result)

    def __predict(self, Xi):
        i, _ = LVQModel.getNearestP(Xi, self.P)
        return self.T[i]


if __name__ == '__main__':
    iris = load_iris()
    data = iris.data
    target = iris.target
    seed = 52514
    np.random.seed(seed)
    np.random.shuffle(data)
    np.random.seed(seed)
    np.random.shuffle(target)

    model = LVQModel()
    model.fit(data, target, 15, max_epoch=50000)
    res = model.predict(data)
    color = np.where(res == 1, 'r', res)
    color = np.where(res == 2, 'g', color)
    color = np.where(res == 0, 'b', color)
    print("F-score: {0:.2f}".format(f1_score(res, target, average='micro')))
    plt.scatter(data[:, 0], data[:, 1], color=color)
    plt.show()
