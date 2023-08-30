import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris


def getNeighbors(X, k, i):
    dist_sort = np.linalg.norm(X - X[i], axis=1).argsort()
    return dist_sort[1:k + 1]


def LLE(data, k, d_=0, t=0.):
    """
    LLE降维, 和sklearn结果不一样
    公式10.28不知道为什么要标一个-1
    :param data: X
    :param k: 近邻参数k
    :param d_: 目标维数d'
    :param t:  同PCA
    :return:   降维后的矩阵
    """
    X = np.array(data)
    m = X.shape[0]

    # 计算ω的矩阵W
    W = np.zeros((m, k))
    for i in range(m):
        neighbors_index = getNeighbors(X, k, i)
        neighbors = X[neighbors_index]

        Cls = np.matmul(X[i] - neighbors, (X[i] - neighbors).T)  # 分母
        for j in range(neighbors_index.shape[0]):
            Cjk = np.matmul(X[i] - X[j], (X[i] - neighbors).T)  # 分子
            W[i][j] = np.sum(Cjk) / np.sum(Cls)

    M = np.matmul(1 - W, (1 - W).T)
    L, Omega = np.linalg.eig(M)

    sort_L_index = np.argsort(L)[::-1]
    if t > 0:
        sumL = np.sum(L)
        threshold = sumL * t
        LSum = 0
        for i, l in enumerate(L):
            LSum += l
            if LSum >= threshold:
                d_ = i + 1
                break

    sort_L_index = sort_L_index[0:d_]
    Omega = Omega[sort_L_index]

    return Omega.T


def main():
    iris = load_iris()
    data = iris.data
    target = iris.target
    data = LLE(data, 3, 2)
    print(data.shape)
    color = np.where(target == 1, 'r', 'm')
    color = np.where(target == 2, 'g', color)
    color = np.where(target == 0, 'b', color)
    plt.scatter(data[:, 0], data[:, 1], color=color)
    plt.show()


if __name__ == '__main__':
    main()
