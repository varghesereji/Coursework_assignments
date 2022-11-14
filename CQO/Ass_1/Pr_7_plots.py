import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import integrate

# for j in range(-2,2,1):
#     for i in range(-3,5,2):
#         Signal = signal.unit_impulse(1000)
#         X = np.linspace(i/4,(i+1)/4,1000) + j*5
#         plt.plot(X,Signal,color='blue')

fig, axs = plt.subplots(2)

axs[0].set_title('Signal')
for x in range(-15,16,2):
    if x%5!=0:
        axs[0].plot([x,x],[0,1],color='blue')
axs[0].plot([-15,15],[0,0],color='blue')

axs[1].set_title('Fourier Transform')
for n in range(-15,16,1):
    axs[1].plot([n/10,n/10],[0,(2/5)*np.cos(3*np.pi*n/10)*np.cos(np.pi*n/10)],color='blue')
axs[1].plot([-1.5,1.5],[0,0],color='blue')
plt.savefig('Pr7_Signal_Transform.png')
