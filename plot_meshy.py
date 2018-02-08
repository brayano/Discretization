import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
import meshy as meshy 
import math
#import othersolvers as tfil
import time

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


def plot_meshy(x,y,m,tune=None, mesh=None,k_interp=None,interp=1, k=1, eps=0.01, xlab='X',ylab='Y',ftrue=None,lnorm=1,ntune=100,extra=False,tag="fig",tf=False):
	# Generating the true function without error for plotting and mse. 
	s = np.linspace(min(x),max(x),3*x.size)
	r = ftrue(s)
	ytrue = ftrue(x)
	
	order = np.argsort(x)
	xs = np.array(x)[order]
	ys = np.array(y)[order]
	
	if k_interp==None:
		k_interp=k
	# Create the coefficients
	if tune != None:
		if tune=='max':
			model = meshy.meshy_one(x,y,k=k, m=m, mesh=mesh,tune = None, eps=eps,lnorm=lnorm)
		else:
			model = meshy.meshy_one(x,y,k=k, m=m, mesh=mesh,tune = tune,eps=eps,lnorm=lnorm)
	else:
		start = time.time()
		models = meshy.meshy(x=x, y=y, m=m, k=k, k_interp=k_interp,interp=interp, mesh=mesh,ftrue=ytrue,lnorm=lnorm,ntune=ntune)
		end = time.time()
		meshytime = end-start
		model = models['minmse.fits']
		tune = models['minmse.lam']
	modelmse = meshy.meshy_mse(model,y=ytrue)
	print "Pars: n=%s, m=%s, r= %s, k= %s, lnorm= %s, lambda = %s, MSE = %s, time = %s" % (len(y), m, k, k_interp, lnorm, tune,modelmse,meshytime)
	fullfits = meshy.meshy_predict(meshy_one_object=model,x=s)

	if extra==False:
		if tf==True:
			start = time.time()
			tfmods = meshy.l1tf(x=x,y=y,k=k,ftrue = ytrue,ntune=ntune)
			end = time.time()
			tftime = end-start
			tfmod = tfmods['minmse.fits']
			print "Pars: n=%s, r= %s, lambda = %s, MSE = %s, time = %s" % (len(y), k, tune,tfmods['minmse'],tftime)
			fulltf = meshy.l1tf_predict(l1tf_one_object=tfmod,x=s)
			plt.figure()
			plt.plot(s,r,'k-')
			plt.vlines(model['mesh'],min(ys)-20*eps,max(ys)+20*eps,alpha=0.5)
			plt.scatter(xs,ys,c='k',alpha=0.1,s=10)
			plt.plot(s,fullfits)
			#plt.plot(s,fulltf,'g-')
			plt.plot(x,tfmod['fitted'],'g-')			
			#plt.legend(bbox_to_anchor=(0.,1.02,1.,.102),borderaxespad=0.,loc=3,ncol=tsize,mode='expand')
			plt.xlabel(xlab)
			plt.ylabel(ylab)
			#plt.show()
			plt.savefig("%s_n%s_m%s_r%s_k%s_l%s.png" % (tag,x.size, m,k,k_interp,lnorm))
		else:
			plt.figure()
			plt.plot(s,r,'k-')
			plt.vlines(model['mesh'],min(ys)-20*eps,max(ys)+20*eps,alpha=0.5)
			plt.scatter(xs,ys,c='k',alpha=0.1,s=10)
			plt.plot(s,fullfits)
			plt.xlabel(xlab)
			plt.ylabel(ylab)
			plt.savefig("%s_n%s_m%s_r%s_k%s_l%s.png" % (tag,x.size, m,k,k_interp,lnorm))
	else :
		plt.figure()
		thetahat = model['theta.hat']
		thetahat = thetahat.reshape((thetahat.size,))
		thetadiffs = np.diff(thetahat)
		nums = np.linspace(1,len(thetadiffs),len(thetadiffs))
		plt.subplot(2,1,1)
		plt.plot(s,r,'k-')
		plt.vlines(model['mesh'],min(r)-20*eps,max(r)+20*eps,alpha=0.5)
		plt.scatter(xs,ys,c='k',alpha=0.1,s=10)
		plt.plot(s,fullfits)
		plt.xlabel(xlab)
		plt.ylabel(ylab)
		
		plt.subplot(2,1,2)
		plt.plot(nums,thetadiffs,'k-')
		plt.ylabel(r"$\theta_{j+1} - \theta_j$")
		plt.xlabel('j')
		plt.savefig("%s_m%s_r%s_k%s_l%s.png" % (tag,m,k,k_interp,lnorm))
		#plt.show()


# Testing
#np.random.seed([117])
#x = np.random.sample(30)
#x = np.linspace(0,1,100)
#x.sort
#def wsin(x):
#	return np.sin(2*math.pi*x)
#y = wsin(x)+np.random.normal(0,1,100)
#plot_meshy(x,y,m=4,mesh=None,tune=None, k_interp=1, k=1, ftrue=wsin,  ntune=50, extra=False,tf=True)

#plot_meshy(x,y,m=10,tune=1000, k_interp=None,interp=1, k=0, ftrue=wsin,  ntune=1, extra=True,tag="sinLam1000")

#plot_meshy(x,y,m=10,tune=20, k_interp=None,interp=1, k=0, ftrue=wsin,  ntune=1, extra=True,tag="sinLam30")

#plot_lambdas(x,y,lambdas=[.001,.01,.05],m=10,ftrue=wsin,k=1,interp=1,lnorm=1)

#plot_lambdas(x,y,lambdas=[.001,.01,.05],m=30,ftrue=wsin,k=2,interp=0,lnorm=2)

