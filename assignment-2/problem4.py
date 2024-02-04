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

#hyperparameters
a=1.0
learning_rate=0.05
iterations= 50

def graph(a,value, color):
    x=np.linspace(0, 1, 100)
    y= sinusoidal(a,x)
    plt.plot(x,y,alpha=0.2*0.9*value, color=color)
graph(a, 1, "green")

def mse(a, x, y):
    y_pred= sinusoidal(a, x)
    mse= (np.mean((y_pred-y)**2))
    return mse
mse_train= np.array([])
mse_train= np.append(mse_train, mse( a, x_train, y_train))


def SGD(a, mse_train):

    for i in range(iterations):
        index= np.random.randint(len(x_train))
        x=x_train[index]
        y=y_train[index]

        y_result= np.sin(a*np.pi*(x)) #predicted y value based on equation
        error= y_result-y

        #derivative
        a-= learning_rate * error * (a* np.pi*np.cos(a*np.pi*x))

        mse_train=np.append(mse_train, mse(a, x_train, y_train))

        value=2
        graph(a, value, "green")
        value+=5
    return mse_train

def SGD_mom(a, mse_train):
    #hyperparameter
    beta=0.75
    v= 0

    for i in range(iterations):
        index= np.random.randint(len(x_train))
        x=x_train[index]
        y=y_train[index]

        y_result= np.sin(a*np.pi*(x)) #predicted y value based on equation
        error= y_result-y

        #derivative
        v= beta *v+ (1-beta)* error * (a* np.pi*np.cos(a*np.pi*x))
        a-= learning_rate * v

        mse_train=np.append(mse_train, mse(a, x_train, y_train))

        value=2
        graph(a, value, "orange")
        value+=5

    return mse_train

def SGD_adam(a, mse_train):
    #hyperparameter
    t=0
    decay_1=0.9
    decay_2=0.999
    epsilon= 0.00000001

    m=0
    v=0

    for i in range (iterations):
        index= np.random.randint(len(x_train))
        x=x_train[index]
        y=y_train[index]

        y_result= np.sin(a*np.pi*(x)) #predicted y value based on equation
        error= y_result-y

        t+=1

        g= error * (a* np.pi*np.cos(a*np.pi*x))
        m= decay_1 * m + (1-decay_1) * g
        v= decay_2 * v + (1-decay_2) * g**2

        m_t= m / (1-decay_1**t)
        #print("m: ", m_t)
        v_t= v/ (1-decay_2**t)
        #print("v: ", v_t)

        a-= (learning_rate*m_t) / (np.sqrt(v_t)+epsilon)
        mse_train=np.append(mse_train, mse(a, x_train, y_train))

        value=2
        graph(a, value, "purple")
        value+=5

    return mse_train

mse_train_sgd= SGD(a, mse_train)
mse_train_sgd_mom= SGD_mom(a, mse_train)
mse_train_sgd_adam= SGD_adam(a, mse_train)

#target function
plt.plot(x_test, y_test, color="black", label="target function")

plt.scatter(x_train,y_train)
plt.title("Gradient Descent")
plt.legend()
plt.show()

plt.plot(range(0,iterations+1), mse_train_sgd, color="green", label="SGD")
plt.plot(range(0,iterations+1), mse_train_sgd_mom, color="orange",  label="SGD_momentum")
plt.plot(range(0,iterations+1), mse_train_sgd_adam, color="purple", label="SGD_adam")


plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("loss vs. epoch")
plt.legend()
plt.show()
