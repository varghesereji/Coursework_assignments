
import matplotlib.pyplot as plt
import numpy as np

def real_en(num, t, a=1, omeg0=10, omega1=1, phi=0):
    return a*np.cos((omega0+num*omega1)*t + phi)
def img_en(num, t, a=1,omeg0=10, omega1=1, phi=0):
    return a*np.sin((omega0+num*omega1)*t + phi)

a=1
omega0 = 10
omega1 = 1
time = np.arange(0,20,0.01)


#Part 1
Waves_re = None
Waves_im = None
for n in range(-5,6,1):
    phi = np.random.uniform(0,2*np.pi,1)
    real_e = real_en(n, time, a,omega0, omega1, phi)
    img_e = img_en(n, time, a,omega0, omega1, phi)
    if Waves_re is None:
        Waves_re = real_e
        Waves_im = img_e
    else:
        Waves_re = np.vstack((Waves_re,real_e))
        Waves_im = np.vstack((Waves_im, img_e))

Real_resultant = np.sum(Waves_re, axis=0)
Imag_resultant = np.sum(Waves_im, axis=0)
Intensity = Real_resultant**2 + Imag_resultant**2

fig = plt.figure(figsize=(16,9))
gs = fig.add_gridspec(3, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle('Amplitude and Intensity in time')
axs[0].set_ylabel('$\Re{E}$')
axs[0].plot(time, Real_resultant, color='blue', label='Real part')
axs[1].set_ylabel('$\Im{E}$')
axs[1].plot(time, Imag_resultant, color='red',label='Imaginary part')
axs[2].set_ylabel('Intensity')
axs[2].plot(time, Intensity,color='green')
axs[2].set_xlabel('time')
plt.legend()
plt.savefig('pr_7_a.png')

#Part 2
Waves_re = None
Waves_im = None
for n in range(-5,6,1):
    phi=0
    real_e = real_en(n, time, a,omega0, omega1, phi)
    img_e = img_en(n, time, a,omega0, omega1, phi)
    if Waves_re is None:
        Waves_re = real_e
        Waves_im = img_e
    else:
        Waves_re = np.vstack((Waves_re,real_e))
        Waves_im = np.vstack((Waves_im, img_e))

Real_resultant = np.sum(Waves_re, axis=0)
Imag_resultant = np.sum(Waves_im, axis=0)
Intensity = Real_resultant**2 + Imag_resultant**2

fig = plt.figure(figsize=(16,9))
gs = fig.add_gridspec(3, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle('Amplitude and Intensity in time')
axs[0].set_ylabel('$\Re{E}$')
axs[0].plot(time, Real_resultant, color='blue', label='Real part')
axs[1].set_ylabel('$\Im{E}$')
axs[1].plot(time, Imag_resultant, color='red',label='Imaginary part')
axs[2].set_ylabel('Intensity')
axs[2].plot(time, Intensity,color='green')
axs[2].set_xlabel('time')
plt.legend()
plt.savefig('pr_7_b.png')
