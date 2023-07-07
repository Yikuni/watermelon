import numpy as np

arr = np.array([[0, 1, 1, 3, 4, 5], [10, 11, 12, 13, 14, 15]])
# print(arr[0, :3])
reshape = arr.reshape(6, 2)
arr.resize(6, 2)
t = arr.T
flatten = arr.flatten()
print(flatten)
# print(t)
# print(reshape)