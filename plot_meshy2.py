import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import scipy as scipy
import meshy as meshy
import meshy2 as meshy2 

# This code is restricted to $x_1,x_2\in [0,1]$

def plot_meshy2(x1,x2,y, m,interp=0, k1=[1],k2=[1], eps=0.001, cvx_solver=0,ftrue=None,mres=50,lnorm=1,ntune=200,xlim=[-1,1],showdata=True):
	xx, yy = np.meshgrid(np.linspace(xlim[0],xlim[1],mres),np.linspace(xlim[0],xlim[1],mres))
	xx_a = xx.reshape((mres**2,1))
	yy_a = yy.reshape((mres**2,1))
	args1 = [xx_a,yy_a]
	args2 = [x1,x2]
	zz = ftrue(args1)
	zz = zz.reshape((mres,mres))
	ytrue = ftrue(args2)
	
	t1 = meshy2.meshy2(x1=x1,x2=x2,y=y,m=m,k1=k1,k2=k2,ftrue=ytrue,interp=interp,lnorm=lnorm,ntune=200)
	fit = t1['minmse.fits']
	tune = t1['minmse.lam']
	modelmse = t1['minmse']
	#fit2 = t1['all_fits'][80]

	xx2, yy2 = np.meshgrid(np.linspace(min(x1)-eps,max(x1)+eps,mres),np.linspace(min(x2)-eps,max(x2)+eps,mres))
	zzhat = meshy2.meshy2_predict(fit,x1=xx2.reshape((mres**2,1))[:,0],x2=yy2.reshape((mres**2,1))[:,0])
	zzhat_grid = zzhat.reshape((mres,mres))
	
	print "Pars: n=%s, m=%s, lambda = %s, MSE = %s" % (len(y), m, tune,modelmse)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(xx, yy, zz, rstride=10, cstride=10)
	ax.plot_surface(xx2,yy2,zzhat_grid,rstride=1,cstride=1)
	if showdata==True:
		ax.scatter(x1,x2,y,s=10)
	plt.show()
	
def plot_meshy2b(x1,x2,y, m,interp=0, k1=1,k2=1, eps=0.01, cvx_solver=0,ftrue=None,mres=50,lnorm=1,ntune=200):
	xx, yy = np.meshgrid(np.linspace(min(x1)-eps,max(x1)+eps,mres),np.linspace(min(x2)-eps,max(x2)+eps,mres))
	xx_a = xx.reshape((mres**2,1))
	yy_a = yy.reshape((mres**2,1))
	args1 = [xx_a,yy_a]
	args2 = [x1,x2]
	zz = ftrue(args1)
	zz = zz.reshape((mres,mres))
	ytrue = ftrue(args2)

	t1 = meshy2.meshy2(x1=x1,x2=x2,y=y,m=m,k1=k1,k2=k2,ftrue=ytrue,interp=interp,lnorm=lnorm,ntune=200)
	fit = t1['minmse.fits']
	tune = t1['minmse.lam']
	modelmse = t1['minmse']
	#fit2 = t1['all_fits'][80]

	xx2, yy2 = np.meshgrid(np.linspace(min(x1),max(x1),mres),np.linspace(min(x2),max(x2),mres))
	zzhat = meshy2.meshy2_predict(fit,x1=xx2.reshape((mres**2,1))[:,0],x2=yy2.reshape((mres**2,1))[:,0])
	zzhat_grid = zzhat.reshape((mres,mres))
	
	print "Pars: n=%s, lambda = %s, MSE = %s" % (len(y), tune,modelmse)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(xx, yy, zz, rstride=10, cstride=10)
	ax.plot_surface(xx2,yy2,zzhat_grid,rstride=1,cstride=1)
	plt.show()	

def plot_meshy2c(x1,x2,y, m,interp=0, k1=1,k2=1, eps=0.01, cvx_solver=0,mres=50,lnorm=1,ntune=200):
	# Data Plotting Function, i.e. no true function is known

	t1 = meshy2.meshy2(x1=x1,x2=x2,y=y,m=m,k1=k1,k2=k2,ftrue=None,interp=interp,lnorm=lnorm,ntune=200)
	fit = t1['minmse.fits']
	tune = t1['minmse.lam']
	modelmse = t1['minmse']
	#fit2 = t1['all_fits'][80]

	xx2, yy2 = np.meshgrid(np.linspace(min(x1),max(x1),mres),np.linspace(min(x2),max(x2),mres))
	zzhat = meshy2.meshy2_predict(fit,x1=xx2.reshape((mres**2,1))[:,0],x2=yy2.reshape((mres**2,1))[:,0])
	zzhat_grid = zzhat.reshape((mres,mres))
	
	print "Pars: n=%s, lambda = %s, MSE = %s" % (len(y), tune,modelmse)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(xx2,yy2,zzhat_grid,rstride=1,cstride=1)
	plt.show()	

# Testing 	
#np.random.seed([117])
#x1 = np.random.uniform(-1,1,50)
#x2 = np.random.uniform(0,2,50)
#y = 2*x1-4*x2+np.random.normal(0,1,50)
#def myplane(args):
#	x= args[0]
#	y = args[1]
#	return 2*x-4*y
#args = [x1,x2]
#ytrue = myplane(args)
#interp=0
#m=20
#mres=50
#k1 = [0,1,0,1]
#k2=[0,0,1,1]
#plot_meshy2b(x1=x1,x2=x2,y=y,m=m,interp=interp,k1=[0],k2=[0],ftrue=myplane)
#plot_meshy2c(x1=x1,x2=x2,y=y,m=m,interp=interp,k1=[0],k2=[0])

def myexp2(args):
	x1 = args[0]
	x2 = args[1]
	z = 2*np.maximum(0,x1+x2)
	y= np.exp(z) - (z+z**2/2+z**3/6)
	return y

np.random.seed([117])
n = 1000
x1 = np.random.uniform(-1,1,n)
x2 = np.random.uniform(-1,1,n)
ytrue = myexp2([x1,x2])
y = ytrue + np.random.normal(0,1,n)
m = 10
plot_meshy2c(x1=x1,x2=x2,y=y,m=m,interp=0,k1=[0],k2=[0])
