import numpy as np
import cvxpy as cvx
import scipy as scipy
import cvxopt as cvxopt
import utils

# First, an l1 trend filter solver:
 
def l1tf_one(x,y,k=1,tune=50,eps=0.01,cvx_solver=0):
	# possible solvers to choose from: typically CVXOPT works better for simulations thus far.  
	solvers = [cvx.SCS,cvx.CVXOPT] # 0 is SCS, 1 CVXOPT
	default_iter = [160000,3200][cvx_solver] # defaults are 2500 and 100

	n = x.size
	# Create D-matrix: specify n and k
	D = utils.form_Dk(n=n,k=k)
	
	# Solve convex problem.
	x = cvx.Variable(n)
	obj = cvx.Minimize(0.5 * cvx.sum_squares(y - x)
                   + tune * cvx.norm(D*x, 1) )
	prob = cvx.Problem(obj)
	prob.solve(solver=solvers[cvx_solver],verbose=False,max_iters = default_iter)

	#print 'Solver status: ', prob.status
	# Check for error.
	#if prob.status != cvx.OPTIMAL:
	#	raise Exception("Solver did not converge!")
	# This while loop only works for SCS. CVXOPT terminates after 1 iteration. 
	counter = 0
	while prob.status != cvx.OPTIMAL:
		maxit = 2*default_iter
		prob.solve(solver=solvers[cvx_solver],verbose=False,max_iters=maxit)
		default_iter = maxit
		counter = counter +1
		if counter>4:
			raise Exception("Solver did not converge with %s iterations! (N=%s,d=%s,k=%s)" % (default_iter,n,ncuts,k) )
	
	output = {'fitted': np.array(x.value),'x':x,'y':y,'eps':eps}  
	return output


def discmse(disco,y):
	yhat = disco['fitted']
	yhat = yhat.reshape((yhat.size,))
	ytrue = y.reshape((y.size,))
	return np.sum((yhat-ytrue)**2)/len(y)


def l1tf(x,y,ftrue=None,k=1,ntune=100,tuners=None,eps=0.01,cvx_solver=0):
	# l1tf is a wrapper on l1tf_one
	# The tuning parameter is not supplied. Instead, l1tf
	# finds the lambda which optimizes MSE, based on either user supplied lambda
	# or number of lambda.
	# ntune must be supplied!

	if tuners == None:
		tuners=np.exp(np.linspace(-5,5,ntune))
	if ftrue == None:
		ftrue = y
	fits = []
	MSEs = []
	for i in range(len(tuners)):
		fits.append(l1tf_one(x=x, y=y,k=k, tune=tuners[i],eps=eps,cvx_solver=cvx_solver))
		MSEs.append(discmse(disco=fits[i],y=ftrue))
	lowestMSE = np.argmin(MSEs)
	
	output = {'minmse.fits':fits[lowestMSE],'minmse':MSEs[lowestMSE],'minmse.lam': tuners[lowestMSE]}
	return output

# Locally Adaptive Regression Splines Solver as described in Ryan Tibshirani (2013) Lemma 1

def knots(x,k):
	x.sort()
	n=x.size
	if k % 2 ==0:
		ind1 = k/2+2; ind2 = n-k/2
	if k % 2 !=0:
		ind1 = (k+1)/2+1; ind2 = n-(k+1)/2
	return  x[(ind1-1):ind2]
		

def tpb(x,t,k):
	G1 = np.zeros((x.size,k))
	for i in range(x.size):
		for j in range(k):
			G1[i,j] = x[i]^j
	
	G2 = np.zeros((x.size,x.size-k-1))
	for i in range(x.size):
		for j in range(x.size-k-1):
			indic = [1 if x[i]>t[j] else 0]
			G2[i,j] = indic*(x[i]-t[j])^k
			
	G = np.concatenate((G1,G2),axis=1)
	return G

	
def lars_one(x,y,k=1,tune=50,eps=0.01,cvx_solver=0):
	# possible solvers to choose from: typically CVXOPT works better for simulations thus far.  
	solvers = [cvx.SCS,cvx.CVXOPT] # 0 is SCS, 1 CVXOPT
	default_iter = [160000,3200][cvx_solver] # defaults are 2500 and 100

	n = x.size

	# Create T: the knots
	T = knots(x=x,k=k)

	# Create G-matrix (truncated power basis): specify n and k
	G = tpb(x=x,t=T,k=k)
	
	# Solve convex problem.

	theta = cvx.Variable(n)
	obj = cvx.Minimize(0.5 * cvx.sum_squares(y - G*theta)
                   + tune * cvx.norm(theta, 1) )
	prob = cvx.Problem(obj)
	prob.solve(solver=solvers[cvx_solver],verbose=False,max_iters = default_iter)

	# Check for error.
	# This while loop only works for SCS. CVXOPT terminates after 1 iteration. 
	counter = 0
	while prob.status != cvx.OPTIMAL:
		maxit = 2*default_iter
		prob.solve(solver=solvers[cvx_solver],verbose=False,max_iters=maxit)
		default_iter = maxit
		counter = counter +1
		if counter>4:
			raise Exception("Solver did not converge with %s iterations! (N=%s,d=%s,k=%s)" % (default_iter,n,ncuts,k) )
	
	output = {'theta.hat':np.array(theta.value),'fitted': G.dot(np.array(theta.value)),'x':x,'y':y,'eps':eps}  
	return output
	
	
def lars(x,y,ftrue=None,k=1,ntune=100,tuners=None,eps=0.01,cvx_solver=0):
	# l1tf is a wrapper on l1tf_one
	# The tuning parameter is not supplied. Instead, l1tf
	# finds the lambda which optimizes MSE, based on either user supplied lambda
	# or number of lambda.
	# ntune must be supplied!

	if tuners == None:
		tuners=np.exp(np.linspace(-5,5,ntune))
	if ftrue == None:
		ftrue = y
	fits = []
	MSEs = []
	for i in range(len(tuners)):
		fits.append(lars_one(x=x, y=y,k=k, tune=tuners[i],eps=eps,cvx_solver=cvx_solver))
		MSEs.append(discmse(disco=fits[i],y=ftrue))
	lowestMSE = np.argmin(MSEs)
	
	output = {'minmse.fits':fits[lowestMSE],'minmse':MSEs[lowestMSE],'minmse.lam': tuners[lowestMSE]}
	return output

