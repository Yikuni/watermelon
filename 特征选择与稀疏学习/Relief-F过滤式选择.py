import numpy as np
from sklearn.datasets import load_iris

'''
Relief-F 过滤式选择
无法过滤掉两个成线性关系的冗余属性
也可以改为用阈值t, 不用目标维度d
用t时具体大小和样本数m, 类别数目γ有关, 可以做一些调整
'''


def getNeighbors(X, Y, i):
    unique_Y = np.unique(Y)
    gamma = unique_Y.shape[0]  # 总共类数
    dist_sort = np.linalg.norm(X - X[i], axis=1).argsort()
    result = np.zeros(gamma)
    for index in dist_sort:
        for l in range(gamma):
            if result[l] == 0 and Y[index] == unique_Y[l]:
                result[l] = index
                break
        if np.all(result != 0):
            break
    return result


def Relief_F(X, Y, d):
    X = np.array(X)
    Y = np.array(Y)
    m = X.shape[0]
    unique_Y = np.unique(Y)
    gamma = unique_Y.shape[0]  # 总共类数
    PL = np.zeros(gamma)

    # 先对X进行规范化到[0, 1]区间
    XMin = np.min(X, axis=0)
    XMax = np.max(X, axis=0)
    X = (X - XMin) / (XMax - XMin)

    # 对每个类别计算Pi
    for i in range(gamma):
        PL[i] = np.sum(np.where(Y == unique_Y[i], 1, 0)) / m

    Delta = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        for i in range(m):
            neighbors = getNeighbors(X, Y, i)
            for l in range(gamma):
                flag = 1
                if unique_Y[l] == Y[i]:
                    flag = -1
                neighbor_index = int(neighbors[l])
                val = (X[i][j] - X[neighbor_index][j]) ** 2
                Delta[j] += flag * val

    # 取绝对值
    Delta = np.abs(Delta)
    print(Delta)
    sort_Delta = np.argsort(Delta)[::-1]
    return sort_Delta[:d]


def main():
    iris = load_iris()
    data = iris.data
    target = iris.target
    col = Relief_F(data, target, 2)
    print("col", col)


if __name__ == '__main__':
    main()
