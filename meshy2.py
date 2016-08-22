import numpy as np
import cvxpy as cvx
import scipy as scipy
import cvxopt as cvxopt
import utils
import utils2
 
	#(cvx.norm(D1*theta*D2.T, lnorm)+cvx.norm(D1*theta,lnorm)/m+cvx.norm(D2*theta.T,lnorm)/m))
def sobbynorm(k1,k2, lnorm,theta,delta1,delta2,m):
	D1 = []
	D2 = []
	norm_list = []
	for i in range(len(k1)):
		D1.append(utils.form_Dk(n=m,k=k1[i])*(delta1**(1/lnorm-k1[i])))
		D2.append(utils.form_Dk(n=m,k=k2[i])*(delta2**(1/lnorm-k2[i])))
		norm_list.append(cvx.norm(D1[i]*theta*D2[i].T,lnorm))
	return np.sum(norm_list)


def meshy2_one(x1,x2,y,m,k1=[1],k2=[1],interp=0,mesh1=None,mesh2=None,lnorm=1,tune=50,eps=0.01,cvx_solver=0):
	# In this bivariate meshy solver, it is assumed that we are taking the same number of cuts per covariate: same m. In general though, can use m_1 cuts for x_1 and m_2 cuts for x_2. 	
	# interp = 0 : nearest neighbor
	# interp = 1 : linear weighting of nearest 2+1=3 neighbors

	# k1 is the order differences applied on x_1, k2 for x_2
	# k1 and k2 can be lists, but they must be the same length such that they are paired at each index

	# possible solvers to choose from: typically CVXOPT works better for simulations thus far.  
	solvers = [cvx.SCS,cvx.CVXOPT] # 0 is SCS, 1 CVXOPT
	default_iter = [160000,3200][cvx_solver] # defaults are 2500 and 100

	n = x1.size

	# Create D-matrix: specify n and k
	if mesh1==None:
		mesh1 = np.linspace(min(x1)-eps,max(x1)+eps,m)
	if mesh2==None:
		mesh2 = np.linspace(min(x2)-eps,max(x2)+eps,m)
	
	delta1 = np.diff(mesh1)[0] # Regular grids, but x1 and x2 can have different range
	delta2 = np.diff(mesh2)[0]

	# Create O-matrix: specify x and number of desired cuts
	O = utils2.interpO(x1=x1,x2=x2,mesh1=mesh1,mesh2=mesh2,interp=interp)
	
	# Solve convex problem.
	theta = cvx.Variable(m,m)
	sobo_like = sobbynorm(k1=k1,k2=k2,lnorm=lnorm,theta=theta,delta1=delta1,delta2=delta2,m=m)
	obj = cvx.Minimize(0.5 * cvx.sum_squares(y - O*cvx.reshape(theta,m*m,1)) + tune * sobo_like)
	prob = cvx.Problem(obj)
	prob.solve(solver=solvers[cvx_solver],verbose=False,max_iters = default_iter)

	counter = 0
	while prob.status != cvx.OPTIMAL:
		maxit = 2*default_iter
		prob.solve(solver=solvers[cvx_solver],verbose=False,max_iters=maxit)
		default_iter = maxit
		counter = counter +1
		if counter>4:
			raise Exception("Solver did not converge with %s iterations! (N=%s,d=%s,k1=%s,k2=%s)" % (default_iter,n,m,k1,k2) )
	
	output = {'mesh1': mesh1, 'mesh2':mesh2,'theta.hat': np.array(theta.value),'fitted':O.dot(np.reshape(np.array(theta.value),m*m,1)),'x1':x1,'x2':x2,'y':y,'k1':k1,'k2':k2,'interp':interp,'eps':eps,'m':m}  
	return output



def meshy2_predict(meshy2_one_object,x1,x2):
	O = utils2.interpO(x1=x1,x2=x2,mesh1=meshy2_one_object['mesh1'],mesh2=meshy2_one_object['mesh2'],interp=meshy2_one_object['interp'])
	m = meshy2_one_object['m']
	return O.dot(np.reshape(meshy2_one_object['theta.hat'],m*m,1))

def meshy2_mse(meshy2_object,y):
	yhat = meshy2_object['fitted']
	yhat = yhat.reshape((yhat.size,))
	ytrue = y.reshape((y.size,))
	return np.sum((yhat-ytrue)**2)/len(y)


def meshy2(x1,x2,y,m,ftrue=None,k1=[1],k2=[1],interp=0,mesh1=None,mesh2=None,lnorm=1,ntune=100,tuners=None,eps=0.01,cvx_solver=0):
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
		fits.append(meshy2_one(x1=x1,x2=x2, y=y, m=m, k1=k1,k2=k2, interp=interp,mesh1=mesh1,mesh2=mesh2, lnorm=lnorm, tune=tuners[i],eps=eps,cvx_solver=cvx_solver))
		MSEs.append(meshy2_mse(meshy2_object=fits[i],y=ftrue))
	lowestMSE = np.argmin(MSEs)
	
	output = {'minmse.fits':fits[lowestMSE],'minmse':MSEs[lowestMSE],'minmse.lam': tuners[lowestMSE],'all_fits':fits}
	return output


#np.random.seed([117])
#x1 = np.random.sample(50)
#x2 = np.random.sample(50)
#y = 2*x1-4*x2+np.random.normal(0,1,50)
#ytrue = 2*x1-4*x2
#t1 = meshy2(x1=x1,x2=x2,y=y,m=10,k1=[0,1,1],k2=[0,0,1],ftrue=ytrue,interp=0)
#t2 = meshy2(x1=x1,x2=x2,y=y,m=30,k1=1,k2=1,ftrue=ytrue,interp=1)
# These next two prints should be the same for k=0,1
#print t1['minmse']
#print t2['minmse']


