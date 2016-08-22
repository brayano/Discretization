import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import scipy as scipy
import meshy as meshy
import meshy2 as meshy2 

# This code is restricted to $x_1,x_2\in [0,1]$

def plot_meshy2(x1,x2,y, m,interp=0, k1=1,k2=1, eps=0.01, cvx_solver=0,ftrue=None,mres=50):
	xx, yy = np.meshgrid(np.linspace(0,1,mres),np.linspace(0,1,mres))
	args1 = [xx,yy]
	args2 = [x1,x2]
	zz = ftrue(args1)
	ytrue = ftrue(args2)

	t1 = meshy2.meshy2(x1=x1,x2=x2,y=y,m=m,k1=k1,k2=k2,ftrue=ytrue,interp=interp)
	fit = t1['minmse.fits']
	#fit2 = t1['all_fits'][80]

	xx2, yy2 = np.meshgrid(np.linspace(min(x1),max(x1),mres),np.linspace(min(x2),max(x2),mres))
	zzhat = meshy2.meshy2_predict(fit,x1=xx2.reshape((mres**2,1))[:,0],x2=yy2.reshape((mres**2,1))[:,0])
	zzhat_grid = zzhat.reshape((mres,mres))
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(xx, yy, zz, rstride=10, cstride=10)
	ax.plot_surface(xx2,yy2,zzhat_grid,rstride=1,cstride=1)
	plt.show()
	


# Testing 	
#np.random.seed([117])
#x1 = np.random.sample(50)
#x2 = np.random.sample(50)
#y = 2*x1-4*x2+np.random.normal(0,1,50)
#def myplane(args):#
#	x= args[0]
#	y = args[1]
#	return 2*x-4*y
#args = [x1,x2]
#ytrue = myplane(args)
#interp=1
#m=20
#mres=50
#plot_meshy2(x1=x1,x2=x2,y=y,m=m,interp=interp,k1=[0,1,0,1],k2=[0,0,1,1],ftrue=myplane)


