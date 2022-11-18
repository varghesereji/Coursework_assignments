import numpy as np
import matplotlib.pyplot as plt


def H0(u):
    return 1


def H1(u):
    return u


def gaussian(u):
    return np.exp(-u**2)


# Circular
r = np.arange(0, 2, 0.01)

gaussr = gaussian(r)
L10 = (1-r)*r

In1 = gaussr**2
In2 = (gaussr*L10)**2
fig, axs = plt.subplots(2, 2, figsize=(16, 9))
axs[0, 1].plot(r, In1, color='blue')
axs[0, 1].plot(-r, In1, color='blue')
axs[0, 1].set_title('Cylindrical $E_{00}$')
axs[1, 1].plot(-r, In2, color='blue')
axs[1, 1].set_title('Cylindrical $E_{10}$')
axs[1, 1].plot(r, In2, color='blue')
axs[0, 1].set_xlabel('r')
axs[0, 1].set_ylabel('Intensity')
axs[1, 1].set_xlabel('r')
axs[1, 1].set_ylabel('Intensity')
# Rectangular

x = np.arange(-2, 2, 0.01)
gauss = gaussian(x)


axs[0, 0].plot(x, gauss**2)
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('Intensity')
axs[1, 0].plot(x, (gauss * H1(x))**2)
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('Intensity')
axs[0, 0].set_title('Rectangular $E_{00}$')
axs[0, 0].set_ylabel('Intensity')
axs[1, 0].set_title('Rectangular $E_{00}$')
axs[1, 0].set_ylabel('Intensity')
plt.savefig('laser_intensity.png')
