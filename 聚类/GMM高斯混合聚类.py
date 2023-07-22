import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

'''
西瓜书上称是Mixture of Gaussian, 所以可能也有其它名字
'''


class GMM:
    def __init__(self, k):
        self.k = k
        self.alpha = None
        self.means = None
        self.sigma = None

    def fit(self, train, max_epoch):
        # 初始化模型参数
        dim = train.shape[1]
        alpha = np.ones(self.k, dtype=float) / self.k
        means = GMM.select_random_vectors(train, self.k)
        _sigma = np.zeros((dim, dim))
        np.fill_diagonal(_sigma, 0.1)
        sigma = []
        for _ in range(self.k):
            sigma.append(_sigma)

        # 计算γji
        m = train.shape[0]
        for _ in range(max_epoch):
            gammas = np.zeros((m, self.k))
            for j in range(m):
                # 分母
                denom = 0
                for l in range(self.k):
                    denom += alpha[l] * GMM.multivariate_normal(train[j], means[l], sigma[l])
                for i in range(self.k):
                    gammas[j, i] = GMM.multivariate_normal(train[j], means[i], sigma[i]) / denom

            # 更新均值向量等
            # μ
            denom = np.sum(gammas, axis=0)
            means = np.dot(gammas.T, train)
            means = means / denom[:, np.newaxis]

            # α
            alpha = np.sum(gammas, axis=0)

            # Σ
            for i in range(self.k):
                diff = train - means[i]
                denom = np.sum(gammas, axis=0)
                sigma[i] = np.matmul(gammas[:, i] * diff.T, diff)
                sigma[i] = sigma[i] / denom[i]

        # 记录结果
        self.alpha = alpha
        self.means = means
        self.sigma = sigma

    def predict(self, x):
        x = np.array(x)
        if len(x.shape) == 1:
            return self.__predict(x)
        else:
            result = []
            for xj in x:
                result.append(self.__predict(xj))
            return np.array(result)

    def __predict(self, xj):
        denom = 0
        gammas = np.zeros(self.k)
        for l in range(self.k):
            denom += self.alpha[l] * GMM.multivariate_normal(xj, self.means[l], self.sigma[l])
        for i in range(self.k):
            gammas[i] = GMM.multivariate_normal(xj, self.means[i], self.sigma[i]) / denom
        lambdaj = np.argmax(gammas)
        return lambdaj

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

    @staticmethod
    def multivariate_normal(x, mu, sigma):
        return np.exp(-0.5 * np.dot(np.dot((x - mu).T, np.linalg.inv(sigma)), (x - mu))) / (
                2 * np.pi * np.sqrt(np.linalg.det(sigma)))


if __name__ == '__main__':
    iris = load_iris()
    data = iris.data

    model = GMM(3)
    model.fit(data, max_epoch=1000)
    res = model.predict(data)
    color = np.where(res == 1, 'r', res)
    color = np.where(res == 2, 'g', color)
    color = np.where(res == 0, 'b', color)
    plt.scatter(data[:, 0], data[:, 1], color=color)
    plt.show()
