import numpy as np


#--------------------------------------------------------------------------------------Hyperparameters
# Network Size
il = 784 # Input layer
h0 = 300 # Hidden layer 0
h1 = 200 # Hidden layer 1
h2 = 100 # Hidden layer 2
ol = 10 # Output layer

lr = 0.0001

# Initializing weights
w0 = np.random.randn(h0, il)*np.sqrt(2/il)
b0 = np.random.randn(h0,  1)*np.sqrt(2/il)
w1 = np.random.randn(h1, h0)*np.sqrt(2/h0)
b1 = np.random.randn(h1,  1)*np.sqrt(2/h0)
w2 = np.random.randn(h2, h1)*np.sqrt(2/h1)
b2 = np.random.randn(h2,  1)*np.sqrt(2/h1)
w3 = np.random.randn(ol, h2)*np.sqrt(2/h2)
b3 = np.random.randn(ol,  1)*np.sqrt(2/h2)


#--------------------------------------------------------------------------------------Activation-function-&-its-derivative
def Softplus(z):
    return np.log(1 + np.exp(z))

def Sigmoid(z):
    return 1/(1 + np.exp(-z))


#--------------------------------------------------------------------------------------Forward-pass
def forward_pass(x):
    z0 = np.dot(w0, x) + b0
    a0 = Softplus(z0)

    z1 = np.dot(w1, a0) + b1
    a1 = Softplus(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = Softplus(z2)

    z3 = np.dot(w3, a2) + b3
    a3 = Sigmoid(z3)

    return a3,z3,a2,z2,a1,z1,a0,z0


#--------------------------------------------------------------------------------------Backward-pass
def backward_pass(e, f_prime, a):
    return lr*np.dot(e*f_prime, a.T), lr*e*f_prime


#--------------------------------------------------------------------------------------Training
with open("C:/Users/pet4r/Desktop/Programing/MnistANNs/mnist_train.csv") as file:
    for data in file:
        x = data.split(',')
        x = np.asfarray(x[1:]).reshape(il, 1) / 255*0.99 + 0.01
        a3,z3,a2,z2,a1,z1,a0,z0 = forward_pass(x)

        t = np.zeros((ol, 1)) + 0.01
        t[int(data[0][0])] = 0.99
        e3 = -(t - a3)
        w,b = backward_pass(e3, a3*(1-a3), a2)
        w3 -= w
        b3 -= b

        e2 = np.dot(w3.T, e3)
        w,b = backward_pass(e2, Sigmoid(z2), a1)
        w2 -= w
        b2 -= b

        e1 = np.dot(w2.T, e2)
        w,b = backward_pass(e1, Sigmoid(z1), a0)
        w1 -= w
        b1 -= b

        e0 = np.dot(w1.T, e1)
        w,b = backward_pass(e0, Sigmoid(z0), x)
        w0 -= w
        b0 -= b


#--------------------------------------------------------------------------------------Testing
positives = 0
with open("C:\\Users\\pet4r\\Desktop\\Programing\\MnistANNs\\mnist_test.csv") as file:
    for data in file:
        x = data.split(',')
        x = np.asfarray(x[1:]).reshape(il, 1) / 255*0.99 + 0.01
        a3,z3,a2,z2,a1,z1,a0,z0 = forward_pass(x)

        if np.argmax(a3) == int(data[0][0]):
            positives += 1

accuracy = (positives/10000)*100
print(accuracy, "%")
