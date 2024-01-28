import numpy as np
from matplotlib import pyplot as plt

from scipy.linalg import svd

mean= np.zeros(3)
cov=[[4,2,1], [2,3,1.5], [1,1.5,2]]

x,y,z= np.random.multivariate_normal(mean, cov, size=1000).T



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)

plt.xlabel("x")
plt.ylabel("y")
plt.show()

U, E, V= svd(cov)
print(U)
print(E)
print(V)