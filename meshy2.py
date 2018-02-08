import numpy as np
import cvxpy as cvx
import cvxopt as cvxopt
import scipy as scipy
import utils
import utils2
import math
import othersolvers as other # Useful for the falling factorial function
import time # USeful for clocking
from scipy.sparse.linalg import spsolve, inv, splu
from scipy.sparse import coo_matrix, vstack, csr_matrix, csc_matrix

def softthresh(z,lam):
	sign  = np.sign(z)
	pmax = np.maximum(abs(z)-lam,0)
	#print type(sign), sign.shape, type(pmax), pmax.shape
	return np.multiply(sign,pmax)

def mypinv(spm, spmt, Oty):
	# spm: sparse matrix; spmt: transpose of sparse matrix
	#return (spmt.dot(inv(spm.dot(spmt)))).dot(spm)
	rows = Oty.shape[0]
	#print spmt.dot(spm)
	a = splu(spmt.dot(spm))
	#print a.shape, Oty.shape, type(Oty)
	b = a.solve(Oty.reshape((rows,)))
	#return inv(spmt.dot(spm)).dot(spmt)
	return spm.dot(b)

def meshy2_one(x1, x2,y, m, theta_init=None,k1=[0],k2=[0], interp=0, mesh1=None, mesh2=None, lnorm=1, tune=1.0, eps=0.01, tol=0.0001, cache=None):
	# An ADMM solver for the bivariate meshy optimization problem with fixed tuning parameter.
	# interp = 0 for nearest neighbor
	# interp = 1 for linear fitting
	# k1,k2 denote the collection of partials for the riemann approximation

	n = x1.size
	y = np.array(y).reshape(n,1)
	
	if cache == None:
		# Create O-matrix: specify x and number of desired cuts
		# Create D-matrix: specify n and k
		O = utils2.interpO(x1=x1,x2=x2,mesh1=mesh1,mesh2=mesh2,interp=interp)
		D = utils.form_D_fvec(m=m,k1=k1,k2=k2,delta1=delta1,delta2=delta2)
		Dt = D.T.tocsc()
		rowsD = D.shape[0]
		cache1 = splu(O.T.dot(O)+tuners[i]*D.T.dot(D))
		cache2 = O.T.tocsc().dot(y)
		if mesh1 is None:
			mesh1 = np.linspace(min(x1)-eps,max(x1)+eps,m)
		if mesh2 is None:
			mesh2 = np.linspace(min(x2)-eps,max(x2)+eps,m)
		
	else:
		cache1 = cache[0]
		cache2 = cache[1]
		D = cache[2]
		Dt = cache[3]
		rowsD = cache[4]
		O = cache[5]
		mesh1 = cache[6]
		mesh2 = cache[7]

	if tune is None:
		A = mypinv(D, Dt, cache2)
		tune = np.max(abs(A))
	
	# Solve convex problem.
	lamda = tune
	rho = tune
	## Initialize
	#print theta_init
	if theta_init is None:
		theta = np.repeat(np.mean(np.array(y)),m**2).reshape(m**2,1)
	else:
		theta = theta_init
	#print D.shape, theta.shape
	alpha = D.dot(theta) #.reshape(m-k-1,1)
	u = np.repeat(1/float(lamda),rowsD).reshape(rowsD,1)
	thetaold = np.repeat(np.mean(np.array(y))-1,m**2).reshape(m**2,1)

	counter = 1
	maxc = 3000
	#times = []

	while any(abs(theta-thetaold)>tol):
	#while (counter<=400):
		thetaold = theta
		alphaold = alpha
		uold = u
		#start = time.time()
		#theta = cache1.dot(O.T.dot(y)+rho*D.T.dot(alpha+u))
		b = (cache2+rho*Dt.dot(alpha+u)).reshape((m**2,))
		theta = (cache1.solve(b)).reshape((m**2,1))
		#theta = spsolve(cache1,O.T.dot(y)+rho*D.T.dot(alpha+u)).reshape((m**2,1))
		#print counter, theta.shape, thetaold.shape	
		#print any(abs(theta-thetaold)>tol)	
		alpha = softthresh(z=D.dot(theta)-u,lam=lamda/float(rho))
		u = u+alpha-D.dot(theta)
		#end = time.time()
		#times.append(end-start)
		counter = counter+1
		if counter>maxc:
			raise Exception("Solver did not converge with %s iterations! (N=%s,m=%s)" % (counter,n,m) )
	#print "For lambda= %s, average cycle time was %s s" % (tune, np.mean(times))
	output = {'mesh1': mesh1, 'mesh2': mesh2,'theta.hat': theta,'fitted':O.dot(theta),'x1':x1,'x2':x2, 'y':y, 'k1':k1, 'k2':k2, 'interp':interp, 'eps':eps,'m':m,'counter': counter} 
	#print counter 
	return output

def meshy2_predict(meshy2_one_object,x1,x2):
	O = utils2.interpO(x1=x1,x2=x2,mesh1=meshy2_one_object['mesh1'],mesh2=meshy2_one_object['mesh2'],interp=meshy2_one_object['interp'])
	m = meshy2_one_object['m']
	return O.dot(meshy2_one_object['theta.hat'])

def meshy2_mse(meshy2_object,y):
	yhat = meshy2_object['fitted']
	yhat = yhat.reshape((yhat.size,))
	ytrue = y.reshape((y.size,))
	return np.sum((yhat-ytrue)**2)/len(y)

def meshy2(x1,x2,y,m,ftrue=None,k1=[1],k2=[1],interp=0,mesh1=None,mesh2=None,lnorm=1,ntune=100,tuners=None,eps=0.01):
	# meshy is a wrapper on meshy_one, where the user supplies only m 
	# The tuning parameter is not supplied. Instead, meshy
	# finds the lambda which optimizes MSE, based on either user supplied lambda
	# or number of lambda. Lambda_max is determined analytically. 
	# ntune must be supplied!
	# Create D-matrix: specify n and k
	n = x1.size
	y = y.reshape((n,1))

	# Create D-matrix: specify n and k	
	if mesh1 is None:
		mesh1 = np.linspace(min(x1)-eps,max(x1)+eps,m)
	if mesh2 is None:
		mesh2 = np.linspace(min(x2)-eps,max(x2)+eps,m)
	
	delta1 = np.diff(mesh1)[0] # Regular grids, but x1 and x2 can have different range
	delta2 = np.diff(mesh2)[0]
	#print mesh1
	#print mesh1[:-1]

	# Create O-matrix: specify x and number of desired cuts
	#start = time.time()
	O = utils2.interpO(x1=x1,x2=x2, mesh1=mesh1, mesh2=mesh2, interp=interp )
	#print O, x1[5], x2[5], mesh1[4], mesh2[4]
	#end = time.time()
	#print "O: %s" % (end-start)
	#start = time.time()
	D = utils.form_D_fvec(m=m,k1=k1,k2=k2,delta1=delta1,delta2=delta2)
	#print D
	#end = time.time()
	#print "D: %s" % (end-start)
	#start = time.time()
	Dt = D.T.tocsc()
	#end = time.time()
	#print "Dt: %s" % (end-start)
	#D2 = csr_matrix(D)
	#D_coo = D.tocoo()
	#D = cvxopt.spmatrix(D_coo.data,D_coo.row.tolist(), D_coo.col.tolist())
	rowsD  = D.shape[0]
	#start = time.time()
	cache2 = O.T.tocsc().dot(y)
	#end = time.time()
	#print "cache2: %s" % (end-start)
	#start = time.time()
	crossD = Dt.dot(D)
	#end = time.time()
	#print "crossD: %s" % (end-start)
	#start = time.time()
	crossO = O.T.tocsc().dot(O)
	#end = time.time()
	#print "crossO: %s" % (end-start)
	#print crack
	if tuners is None:
		#start = time.time()
		A = mypinv(D, Dt, cache2)
		lam_max = np.max(abs(A))		
		#end = time.time()
		#print "Lam_max= %s, Time = %s, s" % (lam_max, end-start)
		#print crack
		tuners=np.exp(np.linspace(np.log(lam_max*10**(-6)),np.log(lam_max),ntune))[::-1]
		#tuners=np.exp(np.linspace(-4,5,ntune))[::-1]
		#tuners=np.exp(np.linspace(np.log(30*10**(-6)),np.log(2),ntune))[::-1]
	
	if ftrue is None:
		ftrue = y
	
	fits = []
	MSEs = []
	counts = []
	thetainit = np.repeat(np.mean(np.array(y)),m**2).reshape(m**2,1)

	for i in range(len(tuners)):
		#start = time.time()
		#cache1 = np.linalg.inv(O.T.dot(O)+tuners[i]*D.T.dot(D))
		#cache1 = splu(O.T.dot(O)+tuners[i]*D.T.dot(D))
		cache1 = splu(crossO+tuners[i]*crossD)
		#print type(O.T.tocsc()), type(D), type(cache1)
		#print crack
		#end = time.time()
		#print "lambda: %s, cache: %s s" % (tuners[i], end-start)
		#cache1 = csr_matrix(O.T.dot(O)+tuners[i]*D.T.dot(D))
		#start = time.time()
		fits.append(meshy2_one(x1=x1, x2=x2,y=y, m=m, k1=k1,k2=k2, interp=interp, mesh1=mesh1, mesh2=mesh2,lnorm=lnorm, tune=tuners[i],eps=eps,theta_init=thetainit,cache=[cache1,cache2,D,Dt,rowsD, O, mesh1, mesh2]))
		#end = time.time()
		#print "Iter %s : %s s; lambda = %s" % (i,end-start, tuners[i])
		#start = time.time()
		MSEs.append(meshy2_mse(meshy2_object=fits[i],y=ftrue))
		#end = time.time()
		#print end-start
		counts.append(fits[i]['counter'])
		thetainit = fits[i]['theta.hat']

	lowestMSE = np.argmin(MSEs)
	#print "ADMM Counts: Min %s, Max %s, Mean %s" % (np.min(counts), np.max(counts), np.mean(counts))
	output = {'minmse.fits':fits[lowestMSE],'minmse':MSEs[lowestMSE],'minmse.lam': tuners[lowestMSE]}
	return output

def myexp2(args):
	x1 = args[0]
	x2 = args[1]
	z = 2*np.maximum(0,x1+x2)
	y= np.exp(z) - (z+z**2/2+z**3/6)
	return y

np.random.seed([117])
n = 100
x1 = np.random.uniform(-1,1,n)
#print x1
x2 = np.random.uniform(-1,1,n)
y = myexp2([x1,x2])+np.random.normal(0,1,n)
#print y
ytrue = myexp2([x1,x2])
m = 5
start = time.time()
t1 = meshy2(x1=x1,x2=x2,y=y,m=m,k1=[0],k2=[0],ftrue=ytrue,interp=0,ntune=100)
end = time.time()
print "m=%s, time = %s s, MSE: %s, minlam = %s" % (m,end-start,t1['minmse'],t1['minmse.lam'])
#print crack

#start = time.time()
#t1 = meshy2(x1=x1,x2=x2,y=y,m=60,k1=[0],k2=[0],ftrue=ytrue,interp=0,ntune=50)
#end = time.time()
#print "m=%s, time = %s s, MSE: %s, minlam = %s" % (60,end-start,t1['minmse'],t1['minmse.lam'])

