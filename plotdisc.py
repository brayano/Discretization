import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
import discsolvers as ds 
	

def dtplot(x,y,vtune,k,ncuts,truefxn,constant=True,eps=0.01,lnorm=1,xlab='X',ylab='Y'):
	# In this function, we plot discrete approximations to 
	# functional minimization problems knowing the truth. 
	# This function is for personal use only.
	# Single number of cuts and k, multiple tuners. 
	
	# Generating the true function without error. 
	s = np.linspace(min(x)-eps,max(x)+eps,400)
	r = truefxn(s)
	
	# determine number of problems to solve
	tsize = len(vtune)
	
	# Create empty list to store fitted values
	#fits = MultiDimList((ncutsize, ksize, tsize))
	fits = []
	
	# Create the coefficients
	for i in range(tsize):
		fits.append( ds.disc_one(x,y,ncuts, k, tune = vtune[i],eps=eps,constant=constant,lnorm=lnorm) )
	
	order = np.argsort(x)
	xs = np.array(x)[order]
	ys = np.array(y)[order]
	segs = np.linspace(min(x)-eps,max(x)+eps,ncuts)
	
	plt.plot(s,r,'k-')
	plt.scatter(xs,ys,c='k',alpha=0.1,s=10)
	plt.vlines(segs,min(y)-eps,max(y)+eps,alpha=0.5)
	if constant==True:
		for j in range(tsize):
			delta = fits[j][0][1]-fits[j][0][0]
			plt.plot(fits[j][0]+delta/2,fits[j][1],label='$\lambda = $ %s' % vtune[j])
	if constant==False:
		for j in range(tsize):
			plt.plot(fits[j][0][order],fits[j][1][order],label='$\lambda = $ %s' % vtune[j])
	plt.legend(bbox_to_anchor=(0.,1.02,1.,.102),borderaxespad=0.,loc=3,ncol=tsize,mode='expand')
	plt.xlabel(xlab)
	plt.ylabel(ylab)
	plt.show()

# Testing 	
#np.random.seed([117])
#x = np.random.normal(0,2,200)
#def wsin(x):
#	return 3*np.sin(3*x)
#y = wsin(x)+np.random.normal(0,1,200)
#fig = plt.figure(figsize=(6, 4))
#dtplot(x,y,vtune=[1,10,30],ncuts=100,truefxn=wsin,k=2,constant=True,lnorm=1)
#fig.savefig("test.png")