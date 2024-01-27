import numpy as np
from scipy import stats

from matplotlib import pyplot as plt

from scipy.stats import multivariate_normal

mean= [5, 6]
cov=[[0.8,0.5], [0.5,1]]

x,y= np.random.multivariate_normal(mean, cov, size=200).T

x_mean = np.mean(x)
x_std = np.std(x)

y_mean = np.mean(y)
y_std = np.std(y)

p=0.5

def bivariate(x,y):
    first =  (1/(2 * np.pi* x_std * y_std * np.sqrt(1-np.power(p,2))))

    z= (np.power((x- x_mean),2)/np.power(x_std,2)) - ((2* p* (x- x_mean)* (y- y_mean))/(x_std* y_std)) + (np.power((y - y_mean),2)/ np.power(y_std,2))

    expression = first * np.exp(-(z/(2*(1-np.power(p,2)))))

    return expression

z= bivariate(x, y)

#print(x_mean, y_mean, x_std, y_std)

plt.tricontour(x,y,z)
plt.scatter(x, y, c=z)

plt.title("Bivariate Normal")
plt.show()