import numpy as np
import matplotlib.pyplot as plt
# import scipy.special as sp

il = 784
h0 = 300
h1 = 100
h2 = 300
ol = 784

m0 = 0
m1 = 0
m2 = 0
m3 = 0
G0 = 0
G1 = 0
G2 = 0
G3 = 0
gamma = 0.9
epsylon = 10**(-8)
lr = 0.0001


w0 = np.random.randn(il, h0)*np.sqrt(2 / il)
b0 = np.random.randn(1,  h0)*np.sqrt(2 / il)
w1 = np.random.randn(h0, h1)*np.sqrt(2 / h0)
b1 = np.random.randn(1,  h1)*np.sqrt(2 / h0)
w2 = np.random.randn(h1, h2)*np.sqrt(2 / h1)
b2 = np.random.randn(1,  h2)*np.sqrt(2 / h1)
w3 = np.random.randn(h2, ol)*np.sqrt(2 / h2)
b3 = np.random.randn(1,  ol)*np.sqrt(2 / h2)

def ReLU(z):
    y1 = z*(z > 0)
    y2 = 0.01*z*(z <= 0)
    return y1 + y2
def ReLU_Prime(z):
    y1 = 1. * (z > 0)
    y2 = 0.01*(z <= 0)
    return y1 + y2

def softplus(z):
    return np.log(1 + np.exp(z))

def expit(z):
    return 1/(1 + np.exp(-z))


def backward_pass(m, G, e, a, f_prime):
    return gamma*m + np.dot(a.T, e*f_prime), G + np.dot(a.T, e*f_prime)**2

file = open("C:\\Users\\pet4r\\Desktop\\Programing\\MnistANN\\mnist_train.csv", "r")
lines = file.readlines()
file.close
#for _ in range(21):
for example in range(60000):
    x = lines[example].split(',')
    x = np.asfarray(x[1:]).reshape((1, il)) / 255 * 0.99 + 0.01

    z0 = np.dot(x, w0) + b0
    a0 = softplus(z0)

    z1 = np.dot(a0, w1) + b1
    a1 = softplus(z1)

    z2 = np.dot(a1, w2) + b2
    a2 = softplus(z2)

    z3 = np.dot(a2, w3) + b3

    e3 = -(x - z3)
    m3,G3 = backward_pass(m3,G3,e3,a2,1)
    w3 -= (lr/(np.sqrt(G3) + epsylon))*m3
    b3 -= lr * e3

    e2 = np.dot(e3, w3.T)
    m2,G2 = backward_pass(m2,G2,e2,a1,expit(z2))
    w2 -= (lr/(np.sqrt(G2) + epsylon))*m2
    b2 -= lr * e2 * expit(z2)

    e1 = np.dot(e2, w2.T)
    m1,G1 = backward_pass(m1,G1,e1,a0,expit(z1))
    w1 -= (lr/(np.sqrt(G1) + epsylon))*m1
    b1 -= lr * e1 * expit(z1)

    e0 = np.dot(e1, w1.T)
    m0,G0 = backward_pass(m0,G0,e0,x,expit(z0))
    w0 -= (lr/(np.sqrt(G0) + epsylon))*m0
    b0 -= lr * e0 * expit(z0)

nocp = 0
file = open("C:\\Users\\pet4r\\Desktop\\Programing\\MnistANN\\mnist_test.csv", "r")
lies = file.readlines()
file.close()
for example in range(10):
        x = lines[example].split(',')
        x = np.asfarray(x[1:]).reshape((1, il)) / 255 * 0.99 + 0.01

        z0 = np.dot(x, w0) + b0
        a0 = softplus(z0)

        z1 = np.dot(a0, w1) + b1
        a1 = softplus(z1)

        z2 = np.dot(a1, w2) + b2
        a2 = softplus(z2)

        z3 = np.dot(a2, w3) + b3

        img = lines[example].split(',')
        img = np.asfarray(img[1:]).reshape((28, 28))
        plt.matshow(img)
        plt.matshow(z3.reshape((28, 28)))

plt.show()
