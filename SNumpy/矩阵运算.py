import numpy as np

grade1 = np.array([90, 86, 74, 59, 77])
grade2 = np.array([92, 84, 74, 51, 72])
grade = np.mat([grade1, grade2]).T
weight = np.array([0.3, 0.7]).T
result = grade.dot(weight)
print(result)
result2 = np.matmul(grade, weight)
print(result2)