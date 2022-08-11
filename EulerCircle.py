import numpy as np
import matplotlib.pyplot as plt

def exp(x):
    return np.sum([x**n/np.math.factorial(n) for n in range(101)])

x = np.linspace(0, 10, 1000)

w = [exp(n*1j) for n in np.mod(x, 2*np.pi)]
a = [n.real for n in w]
s = [n.imag for n in w]

plt.grid()
plt.plot(a, s)
plt.show()
