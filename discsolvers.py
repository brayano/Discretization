import numpy as np
import cvxpy as cvx
import scipy as scipy
import cvxopt as cvxopt
import utils
 
def disc_one(x,y,ncuts,k=1,lnorm=1,tune=50,constant=True,eps=0.01):
	n = x.size
	# Create D-matrix: specify n and k
	segs = np.linspace(min(x)-eps,max(x)+eps,ncuts)
	D = utils.form_Dk(n=ncuts,k=k)

	# Create O-matrix: specify x and number of desired cuts
	O = utils.Omatrix(x,ncuts,constant=constant,eps=eps)
	
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
	if constant==True: return [segs, np.array(theta.value)]
	if constant==False: return [x, O.dot(np.array(theta.value))] 
	

#np.random.seed([117])
#x = np.random.normal(1,5,200)
#y = 2*x**2-4*x-10+np.random.normal(0,1,200)

#print disc_one(x,y,tune=10,ncuts=10)