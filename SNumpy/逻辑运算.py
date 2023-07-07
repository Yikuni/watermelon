import numpy as np

arr = np.array([[0, 1, 1, 3, 4, 5], [10, 11, 12, 13, 14, 15]])
gt10 = arr > 10
result = np.all(gt10)
where = np.where(np.logical_or(arr < 3, arr > 12), 1, 0)
print(where)
