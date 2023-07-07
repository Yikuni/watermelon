import numpy as np
import matplotlib.pyplot as plt
uniform = np.random.uniform(0, 10, 100 )
# print(uniform)
x = np.arange(10000)
y = np.random.normal(0, 1, 100000)
plt.figure()
plt.hist(y, 10000)
plt.show()

