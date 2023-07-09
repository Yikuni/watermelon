import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score

'''
朴素贝叶斯分类器对于属性连续喝离散的数据都可以用
这里分类iris数据集目标的数据是连续的, 于是使用连续的方式
朴素贝叶斯模型
'''


class NaiveBayesModel:
    def __init__(self):
        # P(ci)
        self.classPossibilities = []
        self.means = []
        # σ^2
        self.vars = []
        self.classes = None

    def load(self, path):
        pass

    def save(self, path):
        pass

    @staticmethod
    def calPossibility(mean, var, x: float):
        # x应该是一个数值
        m1 = 1. / np.sqrt(2 * np.pi * var)
        m2 = np.exp(-(x - mean) ** 2 / (2 * var))
        return m1 * m2

    def fit(self, x_train, y_train):
        classes = np.unique(y_train)
        self.classes = classes
        for index, ci in enumerate(classes):
            # 先计算每个类别的概率
            classPossibility = float(np.count_nonzero(y_train == ci)) / len(y_train)
            self.classPossibilities.append(classPossibility)

            # 筛选出ci类中的x然后计算P(xi|ci)
            train_i = x_train[y_train == ci]
            # 再对每个属性计算对应的正态分布参数
            mean = np.mean(train_i, axis=0)
            var = np.var(train_i, axis=0)
            self.means.append(mean)
            self.vars.append(var)

    def __predict(self, x):
        possibilities = []
        for i in range(len(self.classes)):
            mean = self.means[i]
            var = self.vars[i]
            cp = self.classPossibilities[i]
            ps = [NaiveBayesModel.calPossibility(mean[j], var[j], x[j])
                  for j in range(len(x))]
            p = np.prod(ps) * cp
            possibilities.append(p)
        return np.argmax(possibilities)

    def predict(self, x):
        if len(x.shape) == 1:
            return np.array(self.__predict(x))
        elif len(x.shape) == 2:
            return np.array([self.__predict(item) for item in x])


def main():
    iris = load_iris()
    data = iris.data
    target = iris.target
    seed = 52514
    np.random.seed(seed)
    np.random.shuffle(data)
    np.random.seed(seed)
    np.random.shuffle(target)
    x_train, x_test = data[:120, :], data[120:, :]
    y_train, y_test = target[:120], target[120:]

    model = NaiveBayesModel()
    model.fit(x_train, y_train)
    result = model.predict(x_test)
    color = np.where(result == 1, 'r', result)
    color = np.where(result == 2, 'g', color)
    color = np.where(result == 0, 'b', color)
    print("F-score: {0:.2f}".format(f1_score(result, y_test, average='micro')))
    plt.scatter(x_test[:, 0], x_test[:, 1], color=color)
    plt.show()


if __name__ == '__main__':
    main()
