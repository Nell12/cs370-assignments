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

U, E, V= svd(cov)
'''
print(U)
print(E)
print(V)
'''
#Only the first two columns
principal = U[:, :2]

xx, yy = np.meshgrid(np.linspace(-6, 6, 10), np.linspace(-6, 6, 10))
zz = principal[2, 0] * xx + principal[2, 1] * yy
ax.plot_surface(xx, yy, zz, color='yellow', alpha=0.5, label='Projection Plane')

plt.show()