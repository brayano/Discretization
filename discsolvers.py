import numpy as np
import cvxpy as cvx
import scipy as scipy
import cvxopt as cvxopt
import utils
 
def disc_one(x,y,ncuts,k=1,lnorm=1,tune=50,constant=True):
	n = x.size
	# Create D-matrix: specify n and k
	segs = np.linspace(min(x),max(x),ncuts)
	D = utils.form_Dk(n=ncuts,k=k)

	# Create O-matrix: specify x and number of desired cuts
	if constant == True:
		O = utils.form_O(x,ncuts)
	if constant != True:
		O = utils.form_O_linear(x,ncuts)
	
	# Solve convex problem.
	theta = cvx.Variable(ncuts)
	obj = cvx.Minimize(0.5 * cvx.sum_squares(y - O*theta)
                   + tune * cvx.norm(D*theta, lnorm) )
	prob = cvx.Problem(obj)
	# Use CVXOPT as the solver
	prob.solve(solver=cvx.CVXOPT,verbose=False)

	print 'Solver status: ', prob.status
	# Check for error.
	if prob.status != cvx.OPTIMAL:
		raise Exception("Solver did not converge!")
	
	return [segs, np.array(theta.value)]
	
	
def flasso(x,y,tune):
	n = x.size
	order = np.argsort(x)
	x = np.array(x)[order]
	y = np.array(y)[order]
	
	# Create D-matrix
	D = utils.form_Dk(n=n,k=1)
	
	# Solve convex problem.
	theta = cvx.Variable(n)
	obj = cvx.Minimize(0.5 * cvx.sum_squares(y - theta)
                   + tune * cvx.norm(D*theta, 1) )
	prob = cvx.Problem(obj)
	# Use CVXOPT as the solver
	prob.solve(solver=cvx.CVXOPT,verbose=False)

	print 'Solver status: ', prob.status
	# Check for error.
	if prob.status != cvx.OPTIMAL:
		raise Exception("Solver did not converge!")
		
	return [x,theta.value]

#np.random.seed([117])
#x = np.random.normal(1,5,200)
#y = 2*x**2-4*x-10+np.random.normal(0,1,200)

#print disc_one(x,y,tune=10,ncuts=10)