import numpy as np

t1 = np.arange(2, 20,2)
# print(t1)
# print(type(t1))
zeros = np.zeros((3, 4), dtype=np.int32)
ones = np.ones((3, 4), dtype=np.int32)
print(zeros)

copy = np.copy(ones)
ones[0, 2] = 11
# print(copy)

# 生成0, 100范围内, 100个数
linspace = np.linspace(0, 10, 100)
# print(linspace)
