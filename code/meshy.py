import numpy as np
import cvxpy as cvx
import scipy as scipy
import utils
import math
import othersolvers as other # Useful for the falling factorial function
import time # USeful for clocking

def softthresh(z,lam):
	return np.sign(z)*np.maximum(abs(z)-lam,0)

def meshy_one(x, y, m, theta_init=None,k=1, k_interp=None, interp=1, mesh=None, lnorm=1, tune=1.0, eps=0.01, tol=0.001):
	# An ADMM solver for the meshy optimization problem with fixed tuning parameter.
	# interp = 0 for natural splines
	# interp = 1 for banded piecewise polynomials
	# k is the order for the riemann approximation
	# k_interp is the polynomial order interpolation

	n = x.size
	# Create D-matrix: specify n and k
	if mesh==None:
		mesh = np.linspace(min(x)-eps,max(x)+eps,m)
	delta = np.diff(mesh)[0]
	D = utils.form_Dk(n=m,k=k)*(delta**(1/lnorm-k-1))

	# Create O-matrix: specify x and number of desired cuts
	if k_interp==None:
		k_interp= k
	O = utils.interpO(data=x,mesh=mesh,k=k_interp,key=interp)
	
	if tune==None:
		A = O.dot(np.linalg.pinv(D))
		#tune= max(np.max(abs(A.T.dot(y))),1)
		tune = np.max(abs(A.T.dot(y)))
		#print tune
	# Solve convex problem.
	lamda = tune
	rho = tune
	## Initialize
	if theta_init==None:
		theta = np.repeat(np.mean(np.array(y)),m).reshape(m,1)
	else:
		theta = theta_init
	alpha = D.dot(theta) #.reshape(m-k-1,1)
	u = np.repeat(1/float(lamda),m-k-1).reshape(m-k-1,1)
	thetaold = np.repeat(np.mean(np.array(y))-1,m).reshape(m,1)

	counter = 1
	maxc = 3000
	y = np.array(y).reshape(n,1)
	# any(abs(theta-thetanew)>tol)
	while any(abs(theta-thetaold)>tol):
		thetaold = theta
		alphaold = alpha
		uold = u
		
		theta = np.linalg.inv(O.T.dot(O)+rho*D.T.dot(D)).dot(O.T.dot(y)+rho*D.T.dot(alpha+u))
		alpha = softthresh(z=D.dot(theta)-u,lam=lamda/float(rho))
		u = u+alpha-D.dot(theta)
		
		counter = counter+1
		if counter>maxc:
			raise Exception("Solver did not converge with %s iterations! (N=%s,m=%s,r=%s)" % (counter,n,m,k) )
		#if all(abs(theta-thetaold)<tol):
		#	print "Convergence within %s at %s iterations" % (tol,counter-1)
	
	output = {'mesh': mesh, 'theta.hat': theta,'fitted':O.dot(theta),'x':x,'y':y,'k':k,'interp':interp,'k_interp': k_interp, 'eps':eps,'m':m}  
	return output

def meshy(x,y,m,ftrue=None,k=1,k_interp=None,interp=1,mesh=None,lnorm=1,ntune=100,tuners=None,eps=0.01):
	# meshy is a wrapper on meshy_one, where the user supplies only m 
	# The tuning parameter is not supplied. Instead, meshy
	# finds the lambda which optimizes MSE, based on either user supplied lambda
	# or number of lambda. Lambda_max is determined analytically. 
	# ntune must be supplied!
	# Create D-matrix: specify n and k
	if mesh==None:
		mesh = np.linspace(min(x)-eps,max(x)+eps,m)

	delta = np.diff(mesh)[0]
	D = utils.form_Dk(n=m,k=k)*(delta**(1/lnorm-k-1))

	# Create O-matrix: specify x and number of desired cuts
	if k_interp==None:
		k_interp= k
	O = utils.interpO(data=x,mesh=mesh,k=k_interp,key=interp)

	if tuners == None:
		A = O.dot(np.linalg.pinv(D))
		lam_max = np.max(abs(A.T.dot(y))) 
		tuners=np.exp(np.linspace(np.log(lam_max*10**(-6)),np.log(lam_max),ntune))[::-1]
	
	if ftrue == None:
		ftrue = y
	
	fits = []
	MSEs = []
	thetainit = np.repeat(np.mean(np.array(y)),m).reshape(m,1)

	for i in range(len(tuners)):
		fits.append(meshy_one(x=x, y=y, m=m, k=k, k_interp=k_interp, interp=interp, mesh=mesh, lnorm=lnorm, tune=tuners[i],eps=eps,theta_init=thetainit))
		MSEs.append(meshy_mse(meshy_object=fits[i],y=ftrue))
		
		thetainit = fits[i]['theta.hat']

	lowestMSE = np.argmin(MSEs)
	
	output = {'minmse.fits':fits[lowestMSE],'minmse':MSEs[lowestMSE],'minmse.lam': tuners[lowestMSE]}
	return output

def meshy_predict(meshy_one_object,x):
	O = utils.interpO(data=x,mesh=meshy_one_object['mesh'],k=meshy_one_object['k_interp'],key=meshy_one_object['interp'])
	return O.dot(meshy_one_object['theta.hat'])

def meshy_mse(meshy_object,y):
	yhat = meshy_object['fitted']
	yhat = yhat.reshape((yhat.size,))
	ytrue = y.reshape((y.size,))
	return np.sum((yhat-ytrue)**2)/len(y)


def fallfac(xnew, xold,k):
	G1 = np.zeros((xnew.size,k+1))
	for i in range(xnew.size):
		for j in range(k+1):
			G1[i,j] = xnew[i]**j
	
	G2 = np.zeros((xnew.size,xold.size-k-1))
	for i in range(xnew.size):
		for j in range(xold.size-k-1):
			indic = [1 if xnew[i]>=xold[j+k] else 0]
			if indic ==0:
				G2[i,j] = indic
			else:
				storek = np.zeros((k,))
				for l in range(k):
					storek[l] = xnew[i]-xold[j+l+1]
				#print storek			
				G2[i,j] = indic*np.array(np.prod(storek))
	G = np.concatenate((G1,G2),axis=1)
	#print G2
	return G

def l1tf_predict(l1tf_one_object,x):
	H = fallfac(xnew=x,xold=l1tf_one_object['x'],k=l1tf_one_object['k'])
	#print H.shape	
	#print H
	#print H.shape, l1tf_one_object['theta.hat'].shape 
	#print l1tf_one_object['theta.hat']
	fits = H.dot(l1tf_one_object['theta.hat'])
	#print fits
	return fits

def l1tf_mse(l1tf_object,y):
	yhat = l1tf_object['fitted']
	yhat = yhat.reshape((yhat.size,))
	ytrue = y.reshape((y.size,))
	return np.sum((yhat-ytrue)**2)/len(y)

def l1tf_one(x, y, theta_init=None,k=1, tune=1.0, eps=0.01, tol=0.001):
	# An ADMM solver for the l1tf problem with fixed tuning parameter.
	# Need x sorted
	# k is the order Sobolev we approximate
	n = x.size
	
	# Create D-matrix: specify n and k
	D = utils.form_Dk(n=n,k=k)
	if tune==None:
		A = np.linalg.inv(D.dot(D.T))
		tune = np.max(abs((A.dot(D.T)).dot(y)))

	H = other.fallfac(x=x, k=k) # We use H^(-1)theta to get future fitted values
	# Solve convex problem.
	lamda = tune
	rho = tune
	## Initialize
	if theta_init==None:
		theta = np.repeat(np.mean(np.array(y)),n).reshape(n,1)
	else:
		theta = theta_init
	alpha = D.dot(theta) #.reshape(m-k-1,1)
	u = np.repeat(1/float(lamda),n-k-1).reshape(n-k-1,1)
	thetaold = np.repeat(np.mean(np.array(y))-1,n).reshape(n,1)

	counter = 1
	maxc = 3000
	y = np.array(y).reshape(n,1)
	# any(abs(theta-thetanew)>tol)
	while any(abs(theta-thetaold)>tol):
		thetaold = theta
		alphaold = alpha
		uold = u
		
		theta = np.linalg.inv(np.eye(n)+rho*D.T.dot(D)).dot(y+rho*D.T.dot(alpha+u))
		alpha = softthresh(z=D.dot(theta)-u,lam=lamda/float(rho))
		u = u+alpha-D.dot(theta)
		
		counter = counter+1
		if counter>maxc:
			raise Exception("Solver did not converge with %s iterations! (N=%s,m=%s,r=%s)" % (counter,n,m,k) )
		#if all(abs(theta-thetaold)<tol):
		#	print "Convergence within %s at %s iterations" % (tol,counter-1)
	output = {'theta.hat': np.linalg.inv(H).dot(theta),'fitted':theta,'x':x,'y':y,'k':k,'eps':eps}  
	#output = {'theta.hat': H.dot(theta),'fitted':theta,'x':x,'y':y,'k':k,'eps':eps}  
	return output

def l1tf(x,y,ftrue=None,k=1,ntune=100,tuners=None,eps=0.01):
	# meshy is a wrapper on meshy_one, where the user supplies only m 
	# The tuning parameter is not supplied. Instead, meshy
	# finds the lambda which optimizes MSE, based on either user supplied lambda
	# or number of lambda. Lambda_max is determined analytically. 
	# ntune must be supplied!
	# Create D-matrix: specify n and k
	n=x.size
	D = utils.form_Dk(n=n,k=k)
	#print np.linalg.pinv(D).shape
	y = y.reshape(n,1)
	if tuners == None:
		#print D.shape
		A = np.linalg.inv(D.dot(D.T))
		#print A.shape
		lam_max = np.max(abs((A.dot(D)).dot(y)))
		#print lam_max
		tuners=np.exp(np.linspace(np.log(lam_max*10**(-6)),np.log(lam_max),ntune))[::-1]
	
	if ftrue == None:
		ftrue = y
	
	fits = []
	MSEs = []
	thetainit = np.repeat(np.mean(np.array(y)),n).reshape(n,1)

	for i in range(len(tuners)):
		fits.append(l1tf_one(x=x, y=y, k=k, tune=tuners[i],eps=eps,theta_init=thetainit))
		MSEs.append(l1tf_mse(l1tf_object=fits[i],y=ftrue))
		
		thetainit = fits[i]['theta.hat']

	lowestMSE = np.argmin(MSEs)
	
	output = {'minmse.fits':fits[lowestMSE],'minmse':MSEs[lowestMSE],'minmse.lam': tuners[lowestMSE]}
	return output

#np.random.seed([117])
#x = np.linspace(0,1,30)
#x.sort()
#def wsin(x):
#	return np.sin(2*math.pi*x)
#y = wsin(x)+np.random.normal(0,1,30)
#ytrue = wsin(x)
#ytrue = ytrue.reshape(x.size,1)
#k=0
#k_interp=0
#start = time.time()
#test = l1tf(x,y,ftrue=ytrue,k=k)
#end = time.time()
#print test['minmse'], end-start
#m=10
#start = time.time()
#test = meshy(x,y,m,ftrue=ytrue,k=k,k_interp=k_interp)
#end = time.time()
#print test['minmse'], end-start

#m=100
#test = meshy(x,y,m,ftrue=ytrue,k=k,k_interp=k_interp)
#print test['minmse']
