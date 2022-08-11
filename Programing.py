import numpy as np
# import scipy as sp
# from scipy.integrate import quad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import os
import plotly.express as plex


# plt.grid()
# def f(f):
#     F = np.array([np.sin(2*np.pi*n*3)*np.exp(2*np.pi*1j*n*f) for n in t])
#     return [n.real for n in F], [n.imag for n in F]
#
#
# # Define initial parameters
# t = np.linspace(0, 100, 100)
# init_frequency = 1
#
# # Create the figure and the line we will manipulate
# fig, ax = plt.subplots()
# a, s = f(init_frequency)
# line, = plt.plot(a, s, lw = 2)
# ax.set_xlabel('Time [s]')
#
# # Adjust the main plot to make room for the sliders
# plt.subplots_adjust(left = 0.25, bottom = 0.25)
#
# # Make horizontal slider to controll the frequency
# axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
# freq_slider = Slider(
#     ax = axfreq,
#     label = 'Frequency [Hz]',
#     valmin = 0,
#     valmax = 7,
#     valinit = init_frequency
# )
#
# # The function to be called anytime a slider's value changes
# def update(val):
#     a, s = f(freq_slider.val)
#     line.set_xdata(a)
#     line.set_ydata(s)
#     fig.canvas.draw_idle()
#
# # Register the update function with each slider
# freq_slider.on_changed(update)
# plt.show()



# Experimenting with arrays
# a = np.array([[2, 3, 1.7],[3, 1.3, 1], [2.2, 4, 0]])
# b = np.array([
#     [2, 3, 2.2],
#     [3, 1.3, 4],
#     [1.7, 1, 0]
# ])
# print(a)
# print(b)
# print(a.T)
# print(a.reshape(3, 3))

# a = np.array([1.3, 1.7])
# b = np.array([
#     [1.3],
#     [1.7]
# ])
# w0 = np.array([
#     [ 11.88617935, -11.06002727],
#     [ 12.00875365, -11.40469969]
# ])
#
# print(np.dot(a, w0))
# print(a*w0)
# print(a.T*w0)
#--------------------------------------------------------------------------------------
# myShape = np.array([[0, 1, -1, 0], [1, -1, -1, 1]])
# print(myShape.shape)
#
# linearTransform = np.array([
#     [3, 0],
#     [0, 3]
# ])
# rotationTransform = np.array([
#     [np.cos(np.pi/2), -np.sin(np.pi/2)],
#     [np.sin(np.pi/2), np.cos(np.pi/2)]
# ])
#
#
#
# newShape = np.dot(myShape.T,linearTransform).T
# print(newShape)
# print(myShape)
#
# plt.grid()
# plt.plot(myShape[0], myShape[1])
# plt.plot(newShape[0], newShape[1])
# plt.show()


# 2Pi rotation by imaginaryUnit '1j'
# point = np.array(1+0j)
# point1 = np.array(1+0j)*1j
# point2 = np.array(1+0j)*1j**2
# point3 = np.array(1+0j)*1j**3
# point4 = np.array(1+0j)*1j**4
# print(point, point1, point2, point3, point4)
# plt.grid()
# plt.scatter(0, 0)
# plt.scatter(point.real, point.imag)
# plt.scatter(point1.real, point1.imag)
# plt.scatter(point2.real, point2.imag)
# plt.scatter(point3.real, point3.imag)
# plt.scatter(point4.real, point4.imag)
# plt.show()


# 2Pi rotation of a matrix with imaginarUnit '1j'
# myShapeComplex = np.array([
#     [0+1j],
#     [1-1j],
#     [-1-1j],
#     [0+1j]
# ])
# newShape = myShapeComplex*1j**2
#
# plt.grid()
# plt.plot([n.real for n in myShapeComplex], [n.imag for n in myShapeComplex])
# plt.plot([n.real for n in newShape], [n.imag for n in newShape])
# plt.show()

# Rotation of a bunch of poits
# x, y = np.meshgrid(np.arange(-10, 11), np.arange(-10, 11))
# rotationTransform = np.array([
#     [np.cos(np.pi/3), -np.sin((np.pi/3))],
#     [np.sin(np.pi/3), np.cos(np.pi/3)]
# ])
# print(x.shape)
# x = x.reshape(1, 441)
# y = y.reshape(1, 441)
# xy = np.vstack((x, y))
# print(xy.shape)
#
# xy = np.dot(xy.T, rotationTransform).T
# print(xy.shape)
# x = xy[0].reshape(21, 21)
# y = xy[1].reshape(21, 21)
#
#
# plt.grid()
# ax.plot_wireframe(x, y, x*0)
# plt.show()

# NonlinearTransformation?
# def f(x):
#     return x**3
# def polarTransform(x):
#     return
# def softPlus(x):
#     return np.log(1 +  np.exp(x))
# x, y = np.meshgrid(np.linspace(-100, 101), np.linspace(-100, 101))
# y = softPlus(y)
# x = softPlus(x)
#
# ax.plot_wireframe(x, y, x*0)
# plt.show()

# Anaizing ANN
# def softPlus(x):
#     return np.log(1 + np.exp(x))
#
# def sigmoid(x):
#     return 1/(1 + np.exp(-x))
#
# x = np.array([
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ])
# t = np.array([
#     [0],
#     [1],
#     [1],
#     [0]
# ])
#
# w0 = np.array([
#     [ 11.88617935, -11.06002727],
#     [ 12.00875365, -11.40469969]
# ])
# b0 = np.array([
#     [-5.03275633,  9.36240225]
# ])
# w1 = np.array([
#     [-0.66306545],
#     [-1.38433718]
# ])
# b1 = np.array([5.28590777])
# #--------------------------------------------------------------------------------------
# #--------------------------------------------------------------------------------------
# z0 = np.dot(x[0], w0) + b0
# a0 = softPlus(z0)
#
# z1 = np.dot(a0, w1) + b1
# a1 = softPlus(z1)
#
# print(a1)

# qLearning
# qTable = np.zeros((9, 9, 4))
#DQN



# gameSpace = [
#     ['#','#','#','#','#','#','#','#','#'],
#     ['#',' ',' ',' ',' ',' ',' ',' ','#'],
#     ['#',' ',' ',' ',' ',' ',' ',' ','#'],
#     ['#',' ',' ','#',' ','#',' ',' ','#'],
#     ['#',' ',' ','#','O',' ',' ',' ','#'],
#     ['#',' ',' ',' ','#','#',' ',' ','#'],
#     ['#',' ',' ',' ','A',' ',' ',' ','#'],
#     ['#',' ',' ',' ',' ',' ',' ',' ','#'],
#     ['#','#','#','#','#','#','#','#','#']]
#
# epsylon = 1
# for _ in range(101):
#     gameSpace[4][4] = 'O'
#     quit = False
#     r = 0
#     epsylon -= 0.01
#     pAx = cAx = 6
#     pAy = cAy = 4
#     while quit == False:
#         if _ == 100:
#             time.sleep(1)
#         gameSpace[pAx][pAy] = ' '
#
#         if epsylon >= np.random.randint(100)/100:
#             a = np.random.randint(4)
#         else:
#             a = np.argmax(qTable[pAx][pAy])
#
#         if a == 0 and gameSpace[pAx - 1][pAy] != '#':
#             r = -.3
#             cAx = pAx - 1
#             cAy = pAy
#         if a == 1 and gameSpace[pAx][pAy - 1] != '#':
#             r = -.3
#             cAx = pAx
#             cAy = pAy - 1
#         if a == 2 and gameSpace[pAx + 1][pAy] != '#':
#             r = -.3
#             cAx = pAx + 1
#             cAy = pAy
#         if a == 3 and gameSpace[pAx][pAy + 1] != '#':
#             r = -.3
#             cAx = pAx
#             cAy = pAy + 1
#         if a == 'q' or gameSpace[cAx][cAy] == 'O':
#             r = 10
#             quit = True
#         if _ == 100:
#             os.system("cls")
#         gameSpace[cAx][cAy] = 'A'
#         qTable[pAx][pAy][a] = qTable[pAx][pAy][a] + 0.5*(r + 0.99*np.max(qTable[cAx][cAy] - qTable[pAx][pAy][a]))
#         if _ == 100:
#             [print(*n) for n in qTable]
#             [print(*n) for n in gameSpace]
#             print('current', 'past')
#             print([cAx, pAx], [pAx, pAy])
#
#         pAx, pAy = cAx, cAy


# t = np.linspace(0, 1000, 1000)
# f = 3
# plt.grid()
# f = np.array([np.sin(2*np.pi*n*3)*np.exp(2*np.pi*1j*n*f) for n in t])
# a = [n.real for n in f]
# s = [n.imag for n in f]
# plt.plot(a, s)
# plt.plot(x, np.cos(x*2))
# plt.plot(x, np.cos(x*3))
# plt.plot(x, np.cos(x*4))
# plt.plot(x, np.cos(x*5))
# plt.plot(x, np.cos(x*6))
# plt.plot(x, np.cos(2*np.pi*x))
# plt.show()

# def exp(x):
#     return sum([x**n/np.math.factorial(n) for n in range(101)])
# t = np.linspace(0, 4.5, 1000)
# x = np.linspace(0, 200, 1000)
# f = 3
# # f = np.array([np.cos(n*3) for n in np.mod(t, 2*np.pi)])
# # e = np.array([exp(n*3*1j) for n in np.mod(t, 2*np.pi)])
# # w = e*f
# def F(f):
#     F = np.array([np.cos(2*np.pi*n*3)*np.exp(2*np.pi*1j*n*f) for n in t])
#     return[n.real for n in F], [n.imag for n in F]
#
# a, s = F(f)
# plt.grid(linestyle = '-')
# plt.plot(a, s)
# plt.show()

# t = np.linspace(0, 4.5, 1000)
# print(sp.quad(lambda x: np.array([np.cos(2*np.pi*n*3)*np.exp(2*np.pi*1j*n*f) for n in t], 0, 4.5))

# Custom integration

# x = np.linspace(0, 1, 100000)
#
# plt.grid()
# plt.hist([1, 2, 3, 4, 5, 6])
# # plt.hist(x, np.sin(2*np.pi*x))
# plt.show()


x = np.linspace(-30, 30, 700000)

def squareWave(x):
    # x = np.mod(np.pi*x, 2*np.pi)
    if x <= -2:
        return 0
    if x >= 2:
        return 0
    else:
        return 3

def F_squareWave(x):
    return (3*np.sin(2*np.pi*x))/(2*np.pi*x)
    # return np.sin(x*2*np.pi)/(x*2*np.pi)

plt.plot(x, [squareWave(n) for n in x])
# plt.plot(x, [F_squareWave(n) for n in x])
plt.show()

# puno toga mi je drago radit ljepo je bit ziv x)
