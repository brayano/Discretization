import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
import meshy as meshy 

def plot_lambdas(x,y,lambdas,m,k,ftrue,mesh=None,interp=1,eps=0.01,lnorm=1,xlab='X',ylab='Y'):
	# In this function, we plot discrete approximations to 
	# functional minimization problems knowing the truth. 
	# This function is for personal use only.
	# Single number of cuts and k, multiple tuners. 
	
	# Generating the true function without error for plotting and mse. 
	s = np.linspace(min(x)-eps,max(x)+eps,400)
	r = ftrue(s)
	ytrue = ftrue(x)
	
	order = np.argsort(x)
	xs = np.array(x)[order]
	ys = np.array(y)[order]

	# determine number of problems to solve
	tsize = len(lambdas)
	
	# Create empty list to store fitted values
	#fits = MultiDimList((ncutsize, ksize, tsize))
	fits = []
	
	# Create the coefficients
	for i in range(tsize):
		model = meshy.meshy_one(x,y,m, k, interp=interp,mesh=mesh,tune = lambdas[i],eps=eps, lnorm=lnorm)
		modelmse = meshy.meshy_mse(meshy_object = model,y=ytrue)
		print "Pars: n=%s, lambda = %s, MSE = %s" % (len(y), lambdas[i],modelmse)
		fits.append(model)
	
	if mesh==None:
		mesh = np.linspace(min(x)-eps,max(x)+eps,m)
	
	plt.plot(s,r,'k-')
	plt.scatter(xs,ys,c='k',alpha=0.1,s=10)
	plt.vlines(mesh,min(y)-eps,max(y)+eps,alpha=0.5)
	#if k==0:
	#	for j in range(tsize):
	#		delta = fits[j]['mesh'][1]-fits[j]['mesh'][0]
	#		plt.plot(fits[j]['mesh']+delta/2,fits[j]['theta.hat'],label='$\lambda = $ %s' % lambdas[j])
	#else:
	#	for j in range(tsize):
	#		plt.plot(xs,fits[j]['fitted'][order],label='$\lambda = $ %s' % lambdas[j])
	for j in range(tsize):
		full_fitted = meshy.meshy_predict(meshy_one_object=fits[j],x=s)
		plt.plot(s,full_fitted,label='$\lambda = $ %s' % lambdas[j])
	plt.legend(bbox_to_anchor=(0.,1.02,1.,.102),borderaxespad=0.,loc=3,ncol=tsize,mode='expand')
	plt.xlabel(xlab)
	plt.ylabel(ylab)
	plt.show()


def plot_meshy(x,y,m,tune=None, interp=1, k=1, eps=0.01, cvx_solver=0,xlab='X',ylab='Y',ftrue=None):
	# Generating the true function without error for plotting and mse. 
	s = np.linspace(min(x)-eps,max(x)+eps,400)
	r = ftrue(s)
	ytrue = ftrue(x)
	
	order = np.argsort(x)
	xs = np.array(x)[order]
	ys = np.array(y)[order]
	
	# Create the coefficients
	if tune != None:
		model = meshy.meshy_one(x,y,k, tune = tune,eps=eps,cvx_solver=cvx_solver)
	else:
		models = meshy.meshy(x=x,y=y,m=m,k=k,interp=interp,ftrue=ytrue)
		model = models['minmse.fits']
		tune = models['minmse.lam']
	modelmse = meshy.meshy_mse(model,y=ytrue)
	print "Pars: n=%s, lambda = %s, MSE = %s" % (len(y), tune,modelmse)
	fullfits = meshy.meshy_predict(meshy_one_object=model,x=s)

	plt.plot(s,r,'k-')
	plt.scatter(xs,ys,c='k',alpha=0.1,s=10)
	plt.plot(s,fullfits)
	#plt.legend(bbox_to_anchor=(0.,1.02,1.,.102),borderaxespad=0.,loc=3,ncol=tsize,mode='expand')
	plt.xlabel(xlab)
	plt.ylabel(ylab)
	plt.show()


# Testing 	
#np.random.seed([117])
#x = np.random.sample(100)
#def wsin(x):
#	return 3*np.sin(10*x)
#y = wsin(x)+np.random.normal(0,1,100)

#plot_lambdas(x,y,lambdas=[.001,.01,.05],m=30,ftrue=wsin,k=2,interp=1,lnorm=2)

#plot_lambdas(x,y,lambdas=[.001,.01,.05],m=30,ftrue=wsin,k=2,interp=0,lnorm=2)

