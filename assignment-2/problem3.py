import numpy as np
from matplotlib import pyplot as plt
import math

def create_toy_data(func, a, sample_size, std, domain=[0, 1]):
    x = np.linspace(domain[0], domain[1], sample_size)
    np.random.shuffle(x)
    y = func(a, x) + np.random.normal(scale=std, size=x.shape)
    return x, y

def sinusoidal(a,x):
    return np.sin(a * np.pi * x)

x_train, y_train = create_toy_data(sinusoidal, 2, 25, 0.25)
x_test = np.linspace(0, 1, 100)
y_test = sinusoidal(2, x_test)

#target function
plt.plot(x_test, y_test, color="black", label="target function")

#hyperparameters
a=1.0
learning_rate=0.05
iterations= 50

def graph(a,value):
    x=np.linspace(0, 1, 100)
    y= sinusoidal(a,x)
    plt.plot(x,y,alpha=0.2*0.9*value, color="green")
graph(a, 1)

mse_train= np.array([])
def mse(a, x, y):
    y_pred= sinusoidal(a, x)
    mse= (np.mean((y_pred-y)**2))
    return mse

mse_train= np.append(mse_train, mse(a, x_train, y_train))

for i in range(iterations):
    index= np.random.randint(len(x_train))
    x=x_train[index]
    y=y_train[index]

    y_result= np.sin(a*np.pi*(x)) #predicted y value based on equation
    error= y_result-y

    #derivative
    a-= learning_rate * error * (a* np.pi*np.cos(a*np.pi*x))

    mse_train= np.append(mse_train, mse(a, x_train, y_train))

    value=2
    graph(a, value)
    value+=5

#print("MSE: \n",mse_train)

plt.scatter(x_train,y_train)
plt.title("Gradient Descent")
plt.legend()
plt.show()

plt.plot(range(0,iterations+1), mse_train, label="train")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("loss vs. epoch")
plt.legend()
plt.show()