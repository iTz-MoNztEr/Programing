import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

def f(t, amplitude, frequency):
    return amplitude * np.cos(2*np.pi*frequency*t)

# Define initial parameters
t = np.linspace(0, 4.5, 1000)
init_amplitude = 5
init_frequency = 3

# Create the figure and the line we will manipulate
fig, ax = plt.subplots()
line, = plt.plot(t, f(t, init_amplitude, init_frequency), lw = 2)
ax.set_xlabel('Time [s]')

# Adjust the main plot to make room for the sliders
plt.subplots_adjust(left = 0.25, bottom = 0.25)

# Make horizontal slider to controll the frequency
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax = axfreq,
    label = 'Frequency [Hz]',
    valmin = 0.1,
    valmax = 30,
    valinit = init_frequency
)
# Make vertical slider to controll the amplitude
axamp = plt.axes([0.1, 0.25, 0.0225, 0.63])
amp_slider = Slider(
    ax = axamp,
    label = 'Amplitude',
    valmin = 0,
    valmax = 10,
    valinit = init_amplitude
)

# The function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(f(t, amp_slider.val, freq_slider.val))
    fig.canvas.draw_idle()

# Register the update function with each slider
freq_slider.on_changed(update)
amp_slider.on_changed(update)
plt.show()
