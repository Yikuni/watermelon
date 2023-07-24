import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris


'''
不知道是哪里错了, 和sklearn的PCA稍微有点不同
data: 数据集
d_: d', 降维后的维度
t: 重构阈值, d_设置后无效
'''


def PCA(data, d_=0, t=0.):
    X = np.array(data)
    d = X.shape[1]
    if d < d_:
        d_ = 0
    if not (d_ > 0 or t > 0):
        raise Exception("d' <= 0 and t <= 0!")

    # 中心化
    mean = np.mean(data, axis=0)
    X -= mean

    # 计算协方差矩阵X * X.T
    cov = np.matmul(X.T, X)

    # 计算特征值λ和对应特征向量ω的矩阵
    L, Omega = np.linalg.eig(cov)

    # 对特征值进行排序
    sort_L_index = np.argsort(L)[::-1]

    # 如果要根据t计算, 计算出d_
    if t > 0:
        sumL = np.sum(L)
        threshold = sumL * t
        LSum = 0
        for i, l in enumerate(L):
            LSum += l
            if LSum >= threshold:
                d_ = i + 1
                break

    # 用前d'个特征向量组成新矩阵
    sort_L_index = sort_L_index[0:d_]
    Omega_ = Omega[sort_L_index]

    return np.matmul(X, Omega_.T)


def main():
    iris = load_iris()
    data = iris.data
    target = iris.target
    data = PCA(data, 2)
    print(data.shape)
    color = np.where(target == 1, 'r', 'm')
    color = np.where(target == 2, 'g', color)
    color = np.where(target == 0, 'b', color)
    plt.scatter(data[:, 0], data[:, 1], color=color)
    plt.show()


if __name__ == '__main__':
    main()
