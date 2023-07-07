import numpy as np

arr = np.array([[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]])
# 列最大 = arr.max(axis=0)
# 行最大 = arr.max(axis=1)
# print(列最大)
# print(行最大)
var = arr.var(axis=1)
std = arr.std(axis=1)
print(var)
print(std)