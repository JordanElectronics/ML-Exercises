import numpy as np
import matplotlib.pyplot as plt
import math

a = np.array(([3], [1], [2]))
b = a.T
c = a.dot(b)
print(f"{a}\n{b}\n{c}")
    
# a = np.zeros((3, 3))
# b = np.array(([3], [1], [2]))
# c = np.ones((3, 5))
# d = np.arange(1, 10, 0.5)

#print(np.random.rand(5, 3))
#print(np.random.randn(5, 3))

#print(math.sqrt(25))

#plt.plot([1, 2, 3, 4], [2, 3, 2, 0])
#plt.show()
