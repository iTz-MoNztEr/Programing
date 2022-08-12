import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

il = 784
h0 = 300
h1 = 200
h2 = 100
ol = 10

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
lr = 0.001

w0 = np.random.randn(il, h0)*np.sqrt(2 / il)
b0 = np.random.randn(1,  h0)*np.sqrt(2 / il)
w1 = np.random.randn(h0, h1)*np.sqrt(2 / h0)
b1 = np.random.randn(1,  h1)*np.sqrt(2 / h0)
w2 = np.random.randn(h1, h2)*np.sqrt(2 / h1)
b2 = np.random.randn(1,  h2)*np.sqrt(2 / h1)
w3 = np.random.randn(h2, ol)*np.sqrt(2 / h2)
b3 = np.random.randn(1,  ol)*np.sqrt(2 / h2)

def softplus(z):
    return np.log(1+np.exp(z))
'''
def elu(z):
    if z < 0:
        return
'''

def batch():
    x = np.array([])
    for _ in range(300):
        temp = lines[_].split(',')
        temp.append(x)
    return x

def forward_pass(x):
    z0 = np.dot(x, w0) + b0
    a0 = softplus(z0)

    z1 = np.dot(a0, w1) + b1
    a1 = softplus(z1)

    z2 = np.dot(a1, w2) + b2
    a2 = softplus(z2)

    z3 = np.dot(a2, w3) + b3
    a3 = sp.expit(z3)
    return a3,z3,a2,z2,a1,z1,a0,z0
def backward_pass(m, G, a, e, f_prime):
    return gamma*m + np.dot(a.T, e*f_prime), G + np.dot(a.T, e*f_prime)**2

file = open("C:\\Users\\pet4r\\Desktop\\SCIENCE & ART\\SCIENCE & ART\\MyProjects\\MnistANN\\mnist_train.csv", "r")
lines = file.readlines()
file.close

for example in range(60000):
    x = lines[example].split(',')
    x = np.asfarray(x[1:]).reshape((1, il)) / 255 * 0.99 + 0.01
    a3,z3,a2,z2,a1,z1,a0,z0 = forward_pass(x)
    print(np.argmax(a3), [int(lines[example][0])])

    t = np.zeros(ol) + 0.01
    t[int(lines[example][0])] = 0.99
    e3 = -(t - a3)
    m3,G3 = backward_pass(m3,G3,a2,e3, a3 * (1 - a3))
    #w3 -= learning_rate * np.dot(a2.T, (e3 * a3 * (1 - a3)))
    w3 -= (lr/(np.sqrt(G3) + epsylon))*m3
    b3 -= lr * e3 * a3 * (1 - a3)

    e2 = np.dot(e3, w3.T)
    m2,G2 = backward_pass(m2,G2,a1,e2,sp.expit(z2))
    #w2 -= learning_rate * np.dot(a1.T, (e2 * ReLU_Prime(z2)))
    w2 -= (lr/(np.sqrt(G2) + epsylon))*m2
    b2 -= lr * e2 * sp.expit(z2)

    e1 = np.dot(e2, w2.T)
    m1,G1 = backward_pass(m1,G1,a0,e1,sp.expit(z1))
    #w1 -= learning_rate * np.dot(a0.T, (e1 * ReLU_Prime(z1)))
    w1 -= (lr/(np.sqrt(G1) + epsylon))*m1
    b1 -= lr * e1 * sp.expit(z1)

    e0 = np.dot(e1, w1.T)
    m0,G0 = backward_pass(m0,G0,x,e0,sp.expit(z0))
    #w0 -= learning_rate * np.dot(x.T, (e0 * ReLU_Prime(z0)))
    w0 -= (lr/(np.sqrt(G0) + epsylon))*m0
    b0 -= lr * e0 * sp.expit(z0)

nocp = 0
file = open("C:\\Users\\pet4r\\Desktop\\SCIENCE & ART\\SCIENCE & ART\\MyProjects\\MnistANN\\mnist_test.csv", "r")
lies = file.readlines()
file.close()
for example in range(10000):
        x = lines[example].split(',')
        x = np.asfarray(x[1:]).reshape((1, il)) / 255 * 0.99 + 0.01

        a3,z3,a2,z2,a1,z1,a0,z0 = forward_pass(x)
        if np.argmax(a3) == int(lines[example][0]):
            nocp += 1

#        img = lines[example].split(',')
#        img = np.asfarray(img[1:]).reshape((28, 28))
#        plt.matshow(img)
#        plt.show()

accuracy = (nocp / 10000)*100
print(accuracy, '%')
plt.show()
'''
    x = []
    for batch in range(1):
        for example in range(2):
            temp = lines[example].split(',')
            temp = np.asfarray(temp[1:]).reshape((1, il)) / 255 * 0.99 + 0.01
            x.append(temp)
        print(np.shape(x))
        print(x)
        break
        '''
