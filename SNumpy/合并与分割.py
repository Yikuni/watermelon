import numpy as np

grade1 = np.array([[90, 86, 74, 59, 77]])
grade2 = np.array([[92, 84, 74, 51, 72]])

vstack = np.vstack((grade1, grade2))
hstack = np.hstack((grade1, grade2))
print(vstack)
print(hstack)