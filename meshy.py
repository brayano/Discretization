import numpy as np
import cvxpy as cvx
import scipy as scipy
import cvxopt as cvxopt
import utils
 
def meshy_one(x,y,m,k=1,interp=1,mesh=None,lnorm=1,tune=50,eps=0.01,cvx_solver=0):
	# interp = 0 for natural splines
	# interp = 1 for banded piecewise polynomials

	# possible solvers to choose from: typically CVXOPT works better for simulations thus far.  
	solvers = [cvx.SCS,cvx.CVXOPT] # 0 is SCS, 1 CVXOPT
	default_iter = [160000,3200][cvx_solver] # defaults are 2500 and 100

	n = x.size
	# Create D-matrix: specify n and k
	if mesh==None:
		mesh = np.linspace(min(x)-eps,max(x)+eps,m)
	delta = np.diff(mesh)[0]
	D = utils.form_Dk(n=m,k=k)*(delta**(1/lnorm-k))

	# Create O-matrix: specify x and number of desired cuts
	O = utils.interpO(data=x,mesh=mesh,k=k,key=interp)
	
	# Solve convex problem.
	theta = cvx.Variable(m)
	obj = cvx.Minimize(0.5 * cvx.sum_squares(y - O*theta)
                   + tune * cvx.norm(D*theta, lnorm) )
	prob = cvx.Problem(obj)
	prob.solve(solver=solvers[cvx_solver],verbose=False,max_iters = default_iter)

	counter = 0
	while prob.status != cvx.OPTIMAL:
		maxit = 2*default_iter
		prob.solve(solver=solvers[cvx_solver],verbose=False,max_iters=maxit)
		default_iter = maxit
		counter = counter +1
		if counter>4:
			raise Exception("Solver did not converge with %s iterations! (N=%s,d=%s,k=%s)" % (default_iter,n,m,k) )
	
	output = {'mesh': mesh, 'theta.hat': np.array(theta.value),'fitted':O.dot(np.array(theta.value)),'x':x,'y':y,'k':k,'interp':interp,'eps':eps,'m':m}  
	return output


def meshy_predict(meshy_one_object,x):
	O = utils.interpO(data=x,mesh=meshy_one_object['mesh'],k=meshy_one_object['k'],key=meshy_one_object['interp'])
	return O.dot(meshy_one_object['theta.hat'])

def meshy_mse(meshy_object,y):
	yhat = meshy_object['fitted']
	yhat = yhat.reshape((yhat.size,))
	ytrue = y.reshape((y.size,))
	return np.sum((yhat-ytrue)**2)/len(y)


def meshy(x,y,m,ftrue=None,k=1,interp=1,mesh=None,lnorm=1,ntune=100,tuners=None,eps=0.01,cvx_solver=0):
	# meshy is a wrapper on meshy_one, where the user supplies only m 
	# The tuning parameter is not supplied. Instead, meshy
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
		fits.append(meshy_one(x=x, y=y, m=m, k=k, interp=interp,mesh=mesh, lnorm=lnorm, tune=tuners[i],eps=eps,cvx_solver=cvx_solver))
		MSEs.append(meshy_mse(meshy_object=fits[i],y=ftrue))
	lowestMSE = np.argmin(MSEs)
	
	output = {'minmse.fits':fits[lowestMSE],'minmse':MSEs[lowestMSE],'minmse.lam': tuners[lowestMSE]}
	return output


#np.random.seed([117])
#x = np.random.normal(1,2,300)
#y = 2*x**2-4*x-10+np.random.normal(0,1,300)
#ytrue = 2*x**2-4*x-10
#t1 = meshy(x,y,m=50,ftrue=ytrue,k=1,interp=1)
#t2 = meshy(x,y,m=50,ftrue=ytrue,k=1,interp=0)
# These next two prints should be the same for k=0,1
#print t1['minmse']
#print t2['minmse']


