import numpy as np
import matplotlib.pyplot as plt

def softPlus(x):
    return np.log(1 + np.exp(x))

def sigmoid(x):
    return 1/(1 + np.exp(-x))


a = np.arange(-1, 3)
data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
t = np.array([
    [0],
    [1],
    [1],
    [0]
])
# w0 = np.random.uniform(size = (2, 2))
# b0 = np.random.uniform(size = (1, 2))
# w1 = np.random.uniform(size = (2, 1))
# b1 = np.random.uniform(size = (1, 1))
# for n in range(40000):
#     for i in range(4):
#         x = data[i].reshape(1, 2)
#
#         z0 = np.dot(x, w0) + b0
#         a0 = softPlus(z0)
#
#         z1 = np.dot(a0, w1) + b1
#         a1 = softPlus(z1)
#
#         e1 = -(t[i] - a1)
#         w1 = w1 - 0.1*np.dot(a0.T, e1*sigmoid(z1))
#         b1 = b1 - 0.1*e1*sigmoid(z1)
#
#         e0 = np.dot(e1, w1.T)
#         w0 = w0 - 0.1*np.dot(x.T, e0*sigmoid(z0))
#         b0 = b0 - 0.1*e0*sigmoid(z0)
#
#         # print(data[i].reshape(1, 2), a1)
#
# x_intercept = -b0[0][0]/w0[0][0]
# y_intercept = -b0[0][0]/w0[0][1]
# slope = y_intercept/(b0[0][0]/w0[0][0])
#
# o_ = slope*a + y_intercept
# o = o_.reshape(4,)
#
# x_intercept_ = -b0[0][1]/w0[1][0]
# y_intercept_ = -b0[0][1]/w0[1][1]
# slope_ = y_intercept_/(b0[0][1]/w0[1][0])
#
# o__ = slope_*a + y_intercept_
# o_ = o__.reshape(4,)
#
# plt.grid()
# plt.scatter(data[:, 0], data[:, 1], marker = "+", s = 100, c = "black")
# plt.plot(a, o)
# plt.plot(a, o_)
# plt.show()
#
# print(w0)
# print(b0)
# print(w1)
# print(b1)

def ANN_xOR(x):
    w0 = np.array([
        [ 11.88617935, -11.06002727],
        [ 12.00875365, -11.40469969]
    ])
    b0 = np.array([
        [-5.03275633,  9.36240225]
    ])
    w1 = np.array([
        [-0.66306545],
        [-1.38433718]
    ])
    b1 = np.array([5.28590777])

    z0 = np.dot(x, w0) + b0
    a0 = softPlus(z0)

    z1 = np.dot(a0, w1) + b1
    a1 = softPlus(z1)
    return a1
print(ANN_xOR([0, 0]))
print(ANN_xOR([0, 1]))
print(ANN_xOR([1, 0]))
print(ANN_xOR([1, 1]))


print(np.round(ANN_xOR([0, 0])))
print(np.round(ANN_xOR([0, 1])))
print(np.round(ANN_xOR([1, 0])))
print(np.round(ANN_xOR([1, 1])))
