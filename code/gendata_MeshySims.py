import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import scipy as scipy
import math
# This code is restricted to $x_1,x_2\in [0,1]$

def piece_constant(x):
	return -0.6*np.asarray([int(y<=0.3) for y in x]) + 0.8*np.asarray([int(y<=0.75) for y in x]) + 0.5*np.asarray([int(y>0.75) for y in x])

def piece_linear(x):
	return 2*x*np.asarray([int(y<=0.3) for y in x]) -(x-0.9)*np.asarray([int(y>0.3 and y<=0.75) for y in x])+(3*x-(3*.75-.15))*np.asarray([int(y>0.75 and y<=.85) for y in x])+.45*np.asarray([int(y>0.85) for y in x])

def doppler(x):
	y = np.sin(4/x)+1.5
	return y

def mysin(x):
	y = np.sin(2*math.pi*x)+1.5
	return y

def sin_v2(x):
	y = np.sin(4/(x+0.2))
	return y

def myexp(x):
	z = math.pi*x
	y = np.exp(z)
	return y
		
def myexpv2(x):
	z = math.pi*x
	y = np.exp(z)-(1+z+z**2/2+z**3/6)
	return y		

def piece2_constant(args):
	# each list element must be a meshgrid object, mres by mres array
	x1 = args[0]
	x2 = args[1]
	n=x1.size
	y = np.zeros(n)
	for i in range(n): 
		y[i] = 1*np.array(int(x1[i]>0.8))*np.array(int(x2[i]>0.8))+0.5*np.array(int(x1[i]>0.8))*np.array(int(x2[i]<0.2))+1*np.array(int(x1[i]<0.2))*np.array(int(x2[i]<0.2))+0.5*np.array(int(x1[i]<0.2))*np.array(int(x2[i]>0.8))
	return y

def pyramid(args):
	x1= args[0]
	x2 = args[1]
	n = x1.size
	y = np.zeros(n)
	for i in range(n):
		y[i] = 2*x1[i]*np.array(int(x1[i]<x2[i] and x2[i]<1-x1[i]))+2*x2[i]*np.array(int(x1[i]>=x2[i] and x2[i]<1-x1[i]))+(2-2*x2[i])*np.array(int(x1[i]<x2[i] and x2[i]>= 1-x1[i]))+(2-2*x1[i])*np.array(int(x1[i]>=x2[i] and x2[i]>= 1-x1[i]))
	return y

def sombrero(args):
	# This ranges from -inf to +inf
	# For sims, will use -1 to 1
	# each list element must be a meshgrid object, mres by mres array
	x1 = args[0]
	x2 = args[1]
	y = np.sin(15*(x1**2+x2**2)**(0.5))/(25*x1**2+25*x2**2)**(0.5)
	return y

def myexp2(args):
	x1 = args[0]
	x2 = args[1]
	z = 2*np.maximum(0,x1+x2)
	y= np.exp(z)
	return y

def myexp2v2(args):
	x1 = args[0]
	x2 = args[1]
	z = 2*np.maximum(0,x1+x2)
	y= np.exp(z) - (z+z**2/2+z**3/6)
	return y

def gen_univar(x,fxn,error_mu, error_sig):
	return fxn(x)+np.random.normal(error_mu,error_sig,x.size)

def gen_bivar(x1,x2,fxn,error_mu,error_sig):
	args = [x1,x2]	
	y = fxn(args)+ np.random.normal(error_mu, error_sig,x1.size)
	return y


#x = np.linspace(0.01,1,100)
#ytrue = doppler(x)
#y = gen_univar(x,doppler,0,0.5)
#fig = plt.figure()
#plt.xlim(min(x),max(x))
#plt.ylim(min(y),max(y))
#plt.scatter(x, y,c='k',alpha=0.5,s=10)
#plt.plot(x,ytrue)
#plt.show()

##mres = 50
##xx, yy = np.meshgrid(np.linspace(-1,1,mres),np.linspace(-1,1,mres))
#xx, yy = np.meshgrid(np.linspace(0,1,mres),np.linspace(0,1,mres))
#x = np.linspace(0,1,mres)
#xx = np.tile(x,mres)
#y = np.linspace(0,1,mres)
#yy = np.repeat(x,mres)

##args = [xx,yy]
#zz = sombrero(args)
##zz = myexp2(args)
#print zz

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#xx = xx.reshape((mres,mres))
#yy = yy.reshape((mres,mres))
#zz = zz.reshape((mres,mres))
#ax.plot_surface(xx,yy,zz,rstride=1,cstride=1)
#plt.show()


