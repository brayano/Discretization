import numpy as np
import cvxpy as cvx
import scipy as scipy
import cvxopt as cvxopt
import utils
 
def disc_one(x,y,ncuts,k=1,segs=None,lnorm=1,tune=50,constant=True,eps=0.01,cvx_solver=2):
	# possible solvers to choose from: typically CVXOPT works better for simulations thus far.  
	solvers = [cvx.SCS,cvx.ECOS,cvx.CVXOPT] # 0 is SCS, 1 is ECOS, 2 CVXOPT
	
	n = x.size
	# Create D-matrix: specify n and k
	if segs==None:
		segs = np.linspace(min(x)-eps,max(x)+eps,ncuts)
	D = utils.form_Dk(n=ncuts,k=k)

	# Create O-matrix: specify x and number of desired cuts
	O = utils.Omatrix(x,ncuts,constant=constant,eps=eps,segs=segs)
	
	# Solve convex problem.
	theta = cvx.Variable(ncuts)
	obj = cvx.Minimize(0.5 * cvx.sum_squares(y - O*theta)
                   + tune * cvx.norm(D*theta, lnorm) )
	prob = cvx.Problem(obj)
	# Use CVXOPT as the solver
	prob.solve(solver=solvers[cvx_solver],verbose=False)

	print 'Solver status: ', prob.status
	# Check for error.
	if prob.status != cvx.OPTIMAL:
		raise Exception("Solver did not converge!")
	if constant==True: return [segs, np.array(theta.value),x,y,segs,constant,eps]
	if constant==False: return [x, O.dot(np.array(theta.value)),x,y,segs,constant,eps] 
	

def discmse(disco):
	# discmse accepts discretization objects and outputs Mean Squared Error
	# In the disc object (disco), x=object[2], y=object[3],segs = object[4],constant=object[5]
	# eps = disco[6]. disco[0] is eithers segs or x depending on constant=T.
	# disco[1] is either segment estimated coefficients or fitted values, cosntnat=T or F respect. 
	if disco[5]==False:
		MSE = np.sum((disco[1]-disco[3])**2)/len(disco[3])
	if disco[5]==True:
		cats = np.digitize(disco[0], bins=disco[4])
		yhat = []
		for i in range(len(disco[4])):
			yhat.append(disco[1][cats[i]-1])
		MSE = np.sum((disco[3]-yhat)**2)/len(disco[3])
	return MSE
		

def disc_d(x,y,ncuts,k=1,segs=None,lnorm=1,ntune=100,tuners=None,constant=True,eps=0.01,cvx_solver=2):
	# disc_d is a wrapper on disc_one, where the user supplies only d (AKA ncuts)
	# The tuning parameter is not supplied. Instead, disc_d
	# finds the lambda which optimizes MSE, based on either user supplied lambda
	# or number of lambda.
	# ntune must be supplied!
	if tuners == None:
		tuners=np.exp(np.linspace(-4,-5,ntune))
	fits = []
	MSEs = []
	for i in range(len(tuners)):
		fits.append(disc_one(x=x,y=y,ncuts=ncuts,k=k,segs=segs,lnorm=lnorm,tune=tuners[i],constant=constant,eps=eps,cvx_solver=cvx_solver))
		MSEs.append(discmse(fits[i]))
	lowestMSE = np.argmin(MSEs)
	return [fits[lowestMSE], MSEs[lowestMSE],tuners[lowestMSE]]


#np.random.seed([117])
#x = np.random.normal(1,5,200)
#y = 2*x**2-4*x-10+np.random.normal(0,1,200)
#d1object = disc_d(x,y,ntune=10,k=1,ncuts=10,cvx_solver=2,constant=True)
#d2object = disc_one(x,y,tune=10,k=1,ncuts=20,cvx_solver=2,constant=True)
#d3object = disc_one(x,y,tune=10,k=1,ncuts=30,cvx_solver=2,constant=True)
#print np.mean(d1object[0])
#print np.mean(d1object[2])
#print discmse(d1object)
#print discmse(d2object)
#print discmse(d3object)
print d1object[1]

