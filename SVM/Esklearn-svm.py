import numpy as np
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

iris = load_iris()
data = iris.data
target = np.where(iris.target == 0, -1, 1)
seed = 52514
np.random.seed(seed)
np.random.shuffle(data)
np.random.seed(seed)
np.random.shuffle(target)
x_train, x_test = data[:120, :], data[120:, :]
y_train, y_test = target[:120], target[120:]
model = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001,
                cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                break_ties=False, random_state=None)
# 进行训练
model.fit(x_train, y_train)
result = model.predict(x_test)
print(result)
# 进行评估
print("F-score: {0:.2f}".format(f1_score(result, y_test, average='micro')))

plt.scatter(x_test[:, 2], x_test[:, 3], color=np.where(result == -1, 'r', 'g'))
plt.show()
