import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import integrate

# for j in range(-2,2,1):
#     for i in range(-3,5,2):
#         Signal = signal.unit_impulse(1000)
#         X = np.linspace(i/4,(i+1)/4,1000) + j*5
#         plt.plot(X,Signal,color='blue')


X = np.linspace(-4*np.pi,4*np.pi,1000)

Signal = 1-np.absolute(signal.sawtooth(X))




fig, axs = plt.subplots(2)


axs[0].set_title('Signal')
axs[0].plot(X,Signal,color='blue')
axs[0].set_xlabel('x')
axs[1].set_title('Fourier Transform')
for n in range(1,16,1):
    axs[1].plot([n/(2*np.pi),n/(2*np.pi)],[0,2*((-1)**n-1)/n**2],color='blue')
    axs[1].plot([-n/(2*np.pi),-n/(2*np.pi)],[0,2*((-1)**(-n)-1)/n**2],color='blue')

axs[1].plot([-3,3],[0,0],color='blue')
plt.savefig('Pr6_signal_FT.png')
