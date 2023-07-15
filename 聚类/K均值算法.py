import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


class KMeansModel:
    def __init__(self, k):
        self.k = k
        self.result = None

    def fit(self, train):
        # 初始的μ
        means = KMeansModel.select_random_vectors(train, self.k)
        changed = True
        CLA = None
        while changed:
            empty_array = np.empty((0, train.shape[1]), dtype=object)
            CLA = [empty_array.copy() for _ in range(self.k)]
            changed = False
            for x in train:
                c = KMeansModel.getKClass(x, means)
                CLA[c] = np.vstack((CLA[c], x))
            print(CLA)
            for i in range(self.k):
                mean_vector = np.mean(CLA[i], axis=0)
                if not np.array_equal(means[i], mean_vector):
                    means[i] = mean_vector
                    changed = True
        self.result = CLA

    @staticmethod
    def getKClass(xi, means):
        distances = np.linalg.norm(means - xi, axis=1)
        shortest_index = np.argmin(distances)
        return shortest_index

    @staticmethod
    def select_random_vectors(arr, k):
        _arr = np.array(arr)
        # 获取数组的形状
        num_vectors, vector_dim = _arr.shape

        # 生成k个随机索引
        random_indices = np.random.choice(num_vectors, k, replace=False)

        # 根据随机索引选取向量
        random_vectors = _arr[random_indices]

        return random_vectors

    def display(self):
        colors = ['r', 'g', 'b']
        for index, c in enumerate(self.result):
            plt.scatter(c[:, 0], c[:, 1], color=colors[index])
        plt.show()


if __name__ == '__main__':
    data = load_iris().data
    model = KMeansModel(3)
    model.fit(data)
    model.display()
