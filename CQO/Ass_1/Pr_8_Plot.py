import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Cosine_1(fX,fY,n):
    return np.cos(n*np.pi*(fX+fY))
def Cosine_2(fX,fY,n):
    return np.cos(n*np.pi*(fX-fY))

def Sinc(f):
    return np.sin(2*np.pi*f)/(2*np.pi*f)
#Making 3d plot
f_X = np.arange(-5,5,0.1)
f_Y = np.arange(-5,5,0.1)
f_X,f_Y = np.meshgrid(f_X,f_Y)

Delta_1 = 8*(Cosine_1(f_X,f_Y,1)+Cosine_1(f_X,f_Y,3)+Cosine_1(f_X,f_Y,5)+Cosine_1(f_X,f_Y,7))
Delta_2 = 16*(Cosine_1(f_X,f_Y,1)+Cosine_1(f_X,f_Y,3)+Cosine_1(f_X,f_Y,5))*Cosine_2(f_X,f_Y,2)
Delta_3 = 16*(Cosine_1(f_X,f_Y,1)+Cosine_1(f_X,f_Y,3))*(Cosine_2(f_X,f_Y,2)+Cosine_2(f_X,f_Y,4))
Delta_4 = 16*(Cosine_1(f_X,f_Y,1))*(Cosine_2(f_X,f_Y,2)+Cosine_2(f_X,f_Y,4)+Cosine_2(f_X,f_Y,6))

Function = (Delta_1+Delta_2+Delta_3+Delta_4)*Sinc(f_X)*Sinc(f_Y)


fig = plt.figure(figsize=(16,9))
ax = plt.axes(projection='3d')
ax.grid()
ax.plot_surface(f_X,f_Y,Function)
plt.xlabel('f_X')
plt.ylabel('f_Y')
ax.set_title('')
plt.show(block=False)
plt.savefig('3D_plot_pr8.png')
