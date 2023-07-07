import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
data = iris.data
target = iris.target
beta = np.zeros(5, dtype=np.double)  # 第0列是b, 第一列是w
ones = np.ones((150, 1))
X = np.hstack((ones, data))
Y = np.where(target == 0, 0, 1)  # 0为0, 1和2为1, OvR多分类


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_model(iterations=3):
    global beta
    for i in range(iterations):
        p1 = 1 - sigmoid(X.dot(beta.T))
        p = np.diag((p1 * (1 - p1))).T
        first_derivative = -X.T.dot(Y - p1)
        second_derivative = X.T.dot(p).dot(X)
        inv = np.linalg.inv(second_derivative)
        beta -= inv.dot(first_derivative.T)


if __name__ == '__main__':
    build_model(3)
    predicts_prob = sigmoid(X.dot(beta.T))
    print(beta)
    colors = np.where(predicts_prob < 0.5, 'r', 'g')
    plt.scatter(data.T[0], data.T[1], color=colors)
    plt.show()
