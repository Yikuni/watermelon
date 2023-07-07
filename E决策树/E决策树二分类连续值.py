import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

'''
决策树二分类预剪枝
'''
iris = datasets.load_iris()
data = iris.data
target = iris.target
target = np.where(target == 2, 1, target)

seed = 514
np.random.seed(seed)
np.random.shuffle(data)
np.random.seed(seed)
np.random.shuffle(target)

train_length = int(data.shape[0] * 4 / 5)
x_train = np.array(data[:train_length])
x_test = np.array(data[train_length:])
y_train = np.array(target[:train_length])
y_test = np.array(target[train_length:])

max_depth = 8


class TreeNode:
    def __init__(self, y_label, dim, t, parent=None):
        self.parent = parent
        self.y_label = y_label
        self.dim = dim
        self.t = t
        self.left = None
        self.right = None

    def predict(self, arr):
        if arr[self.dim] >= self.t:
            if self.right is None:
                return self.y_label
            return self.right.predict(arr)
        else:
            if self.left is None:
                return self.y_label
            return self.left.predict(arr)


class DTree:
    def __init__(self):
        self.root = None

    def fit(self, x_train, y_train, x_test, y_test):
        acc_history = []

        def gen_node(parent, x, y, label):
            p = parent
            depth = 0
            while p is not None:
                depth += 1
                p = p.parent
                if depth >= max_depth:
                    print("reach max depth")
                    return
            t_and_gains = []
            for i in range(x.shape[1]):
                t_and_gains.append(DTree.cal_gain(x, y, i))
            arr_t_and_gains = np.array(t_and_gains)
            dim = arr_t_and_gains.argmax(axis=0)[1]
            t = t_and_gains[dim][0]
            left_index = np.where(x < t)
            left_index = np.unique(left_index)
            left_x = np.array(x[left_index])
            left_y = np.array(y[left_index])
            right_index = np.unique(np.where(x < t))
            right_x = np.array(x[right_index])
            right_y = np.array(y[right_index])
            left_predict = 0
            right_predict = 1
            if len(left_y) < 1 or len(right_y) < 1:
                return
            if left_y.sum() / left_y.shape[0] > right_y.sum() / right_y.shape[0]:
                left_predict = 1
                right_predict = 0
            acc_before = self.cal_acc(x_test, y_test)
            node = TreeNode(label, dim, t, parent)

            if parent is None:
                self.root = node
            else:
                if label == 0:
                    parent.left = node
                else:
                    parent.right = node

            # 预剪枝
            acc_after = self.cal_acc(x_test, y_test)
            if acc_before >= acc_after:
                if label == 0:
                    parent.left = None
                else:
                    parent.right = None
                return
            acc_history.append(acc_after)
            gen_node(node, left_x, left_y, left_predict)
            gen_node(node, right_x, right_y, right_predict)

        gen_node(None, x_train, y_train, 1)
        plt.plot(np.arange(len(acc_history)) + 1, acc_history, 'g')
        plt.show()

    def predict(self, arr):
        return self.root.predict(arr)

    def cal_acc(self, x_test, y_test):
        if self.root is None:
            return 0
        predict_y = []
        for element in x_test:
            predict_y.append(self.root.predict(element))
        tp = np.array(np.where(predict_y == y_test)).shape[1]
        return tp / y_test.shape[0]

    @staticmethod
    def cal_ent(y):
        p = np.where(y == y[0], 1, 0).sum() / len(y)
        ent = -p * np.log(p) - (1 - p) * np.log(1 - p)
        return ent

    # 计算信息增益, 返回[t, gain]
    @staticmethod
    def cal_gain(x, y, dim):
        if len(y) < 2:
            return [0, 0]
        dim_x = x[:, dim]
        sorted_x = np.sort(dim_x)
        t_and_gain = []
        for i in range(len(dim_x) - 1):
            t = (sorted_x[i] + sorted_x[i + 1]) / 2
            d_len = len(y)
            dv1 = np.array(np.where(dim_x < t))
            dv2 = np.array(np.where(dim_x >= t))
            if dv1.shape[1] < 2 or dv2.shape[1] < 2:
                continue
            gain = DTree.cal_ent(y) - dv1.shape[1] * DTree.cal_ent(np.array(y)[dv1][0]) / d_len
            gain = gain - dv2.shape[1] * DTree.cal_ent(np.array(y)[dv2][0]) / d_len
            if np.isnan(gain):
                continue
            t_and_gain.append([t, gain])
        if not t_and_gain:
            return [0, 0]
        arr_t_and_gain = np.array(t_and_gain)
        index = arr_t_and_gain.argmax(axis=0)[1]

        return t_and_gain[index]


dTree = DTree()
dTree.fit(x_train, y_train, x_test, y_test)
result = []
for i in range(y_test.shape[0]):
    result.append(dTree.predict(x_test[i]))
print(result)
print(y_test)
print(np.where(result == y_test, 1, 0))
