import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix, csc_matrix, vstack, linalg as sla
import cvxopt as cvxopt
import math
import itertools
from operator import add, sub

def form_D1(n):
	D = np.zeros((n-1,n))
	for i in range(n-1):
		D[i,i] = 1
		D[i,i+1] = -1
	return D 

def form_Dk(n,k):
	D = np.identity(n)
	for i in range(k+1):
		D = form_D1(n-i).dot(D)
	return D

def form_O(x,ncuts,eps=0.01,segs=None):
	if segs==None:
		segs = np.linspace(min(x)-eps,max(x)+eps,ncuts)
	# Default: bins[i-1]<= x < bins[i]
	cats = np.digitize(x, bins=segs)
	n = x.size
	O = np.zeros((n, ncuts))
	for i in range(n):
		O[i,cats[i]-1] = 1
	return O

def form_O_linear(x,ncuts,eps=0.01,segs=None):
	# Produce a linear interpolation knot matrix. 
	# Unless segments are provided, we assume equal bin
	# sizes. 
	if segs==None:
		segs = np.linspace(min(x)-eps,max(x)+eps,ncuts)
	cats = np.digitize(x, bins=segs)
	delta = np.ediff1d(segs) # length of each bin
	delta = np.append(delta,delta[-1]+eps)

	# Calculate the distance between a point and the beginning
	# of its corresponding bin. 
	t = x-segs[cats-1] 
	n = x.size
	O = np.zeros((n, ncuts))
	for i in range(n):
		index = cats[i] - 1
		O[i,index] =  (delta[index]-t[i])/delta[index]
		if index+1 < ncuts:
			O[i,index+1] = t[i]/delta[index+1]
	return O

def Omatrix(x,ncuts,constant=True,eps=0.01,segs=None):
	# Currently, Omatrix outputs either constant knot matrix
	# or linearly interpolated knot matrix. Eventually,
	# "constant" will be deprecated for "degree" option.
	# degree=0; constant. degree=1; linear. degree=2; quadratic. etc. 
	if constant == True:
		O = form_O(x,ncuts,eps=eps,segs=segs)
	if constant != True:
		O = form_O_linear(x,ncuts,eps=eps,segs=segs)
	return O

## Create functions for the banded piecewise polynomials and natural splines

# Natural Splines

def knots(x,k):
	# From a mesh of m points, splines will occur over m-2(k+1) of the knots
	x.sort()
	n=x.size
	if k % 2 ==0:
		ind1 = k/2+2; ind2 = n-k/2
	if k % 2 !=0:
		ind1 = (k+1)/2+1; ind2 = n-(k+1)/2
	return  x[(ind1-1):(ind2)]
	

def designNS(x,mesh,k):
	t = knots(x=mesh,k=k)
	G1 = np.zeros((x.size,k+1))
	for i in range(x.size):
		for j in range(k+1):
			G1[i,j] = x[i]**j
	G2 = np.zeros((x.size,t.size))
	for i in range(x.size):
		for j in range(t.size):
			indic = [1 if x[i]>=t[j] else 0]
			G2[i,j] = indic*np.array((x[i]-t[j])**k)	
	G = np.concatenate((G1,G2),axis=1)
	return G

def interpNS(data,mesh,k):
	Psi = designNS(x=mesh,mesh=mesh,k=k)
	psitilde = designNS(x=data,mesh=mesh,k=k)
	return psitilde.dot(np.linalg.inv(Psi))

# Banded Piecewise Polynomials

# For a single observation
def designBPP(x, k):
	G = np.zeros((x.size,k+1))
	for i in range(x.size):
		for j in range(k+1):
			G[i,j] = x[i]**j
	return G

def interpBPP(data,mesh,k):
	inds = np.digitize(data,mesh)-1
	psitilde = designBPP(x=data,k=k)
	O = np.zeros((data.size,mesh.size))
	for i in range(data.size):
		if inds[i] >= mesh.size-1-k:
			Psi = designBPP(x=mesh[np.arange(mesh.size-1-k,mesh.size)],k=k)
			O[i,(mesh.size-1-k):(mesh.size)] = psitilde[i,].dot(np.linalg.inv(Psi))
		else:
			Psi = designBPP(x=mesh[np.arange(inds[i],inds[i]+k+1)],k=k)
			O[i,inds[i]:(inds[i]+k+1)] =  psitilde[i,].dot(np.linalg.inv(Psi))
	return O

def interpO(data,mesh,k,key):
	# key=0: NS, key=1: BPP
	if key == 0:
		O = interpNS(data,mesh,k)
	elif key == 1:
		O = interpBPP(data,mesh,k)
	else:
		raise Exception("Not a solver key! Only Key=0 (NS),1 (BPP)")
	return O

# Penalty Approximations

def form_D_fvec_ft(m,k):
	# Let a cardinal difference vector, c, denote the relevant rth order difference vector
	# i.e. for 1st order difference, c= (1,-1). For 2nd order difference, c = (1,-2,1). Let theta= (theta_1, theta_2, ..., theta_c) denote relevant thetas for a difference segment. 
	n=k+2
	c = form_Dk(n,k).reshape((1,n)) # cardinal difference vector
	co = (k+1)*m+1 # In an (mxm)-array, if we index them sequentially row-wise, then the column wise first order differences are m-indices different. For second order differences column wise, theta_2 is m-away from theta_1, theta_3 is m-away from theta_2. Hence, we need a cardinal open vector (covec) with 2*m+1 many elements. 
	covec = np.zeros((1,co))
	for i in range(n):
		covec[0,i*m] = c[0,i]
	# m(m-k-1) is the number of unique rows we need to define all differences. 
	step = np.zeros((m*(m-k-1),m**2))
	for i in range(m*(m-k-1)):
		step[i,i:(co+i)] = covec
	return step

def form_D_fvec_f(m,k):
	# This function returns Difference matrix for vectorized f when 
	# attempting D
	n=k+2
	blocks = []
	D = form_Dk(m,k)
	for i in range(m):
		blocks.append(D)
	return block_diag( *blocks)
	
def form_D_fvec_cross(m,k1,k2):
	# k1>=k2
	n1=k1+2
	n2=k2+2
	c1 = form_Dk(n1,k1).reshape((1,n1))
	c2 = form_Dk(n2,k2).reshape((1,n2))
	myn = n1*n2+(n2-1)*(m-k1-2)
	covec = np.zeros((1,myn))
	#print myn
	for i in range(n2):
		#print i*(n1+m-k1-2),i*(n1+m-k1-2)+n1
		#covec[0,i*(n2+m-k1-2+1-k2):i*(n2+m-k1-2+1-k2)+n1] = c2[0,i]*c1
		covec[0,i*(n1+m-k1-2):i*(n1+m-k1-2)+n1] = c2[0,i]*c1
	step = np.zeros((m**2-myn+1,m**2))
	for i in range(m**2-myn+1):
		#print i
		step[i,i:(myn+i)] = covec
	return step

def form_D_fvec(m,delta1,delta2,k1=[0],k2=[0]):
	s = len(k1)
	for i in range(s):
		D1 = form_D_fvec_f(m=m,k=k1[i])*(delta1**(1-k1[i])) # removing -1 from exponent
		D2 = form_D_fvec_ft(m=m,k=k2[i])*(delta2**(1-k2[i]))
		D3 = form_D_fvec_cross(m=m,k1=k1[i],k2=k2[i])*(delta1**(1-k1[i]))*(delta2**(1-k2[i]))
	#print D1.shape, D2.shape, D3.shape
	#D = np.concatenate((D1,D2,D3),axis=0)
	#D = vstack([D1,D2,D3])	
	D = csc_matrix(vstack([D1,D2,D3]).toarray())
	#return cvxopt.spdiag([D])
	#print type(D)
	return D



#print form_D_fvec(m=5)
#d = np.linspace(0,1,10)
#x = np.random.sample(40)
#print x
#Psi = designNS(x=d,mesh=d,k=3)
#psitilde = designNS(x=x,mesh=d,k=3)
#print psitilde.dot(np.linalg.inv(Psi))
#Psi = designBPP(x=d[np.arange(1,3)],k=1)
#print d[np.arange(1,2)]
#print designBPP(x,k=1)[1,].dot(np.linalg.inv(Psi))
#print interpNS(data=x,mesh=d,k=0)
#print interpBPP(data=x,mesh=d,k=2)
#interpO(data=x,mesh=d,k=1,key=2)

#np.random.seed([117])
#x = np.random.normal(0,2,40)
#def wsin(x):
#	return 3*np.sin(3*x)
#y = wsin(x)+np.random.normal(0,1,40)
#print interpBPP(data=x,mesh=np.linspace(min(x)-.01,max(x)+.01,10),k=2)

#d = np.linspace(0,2,10)
#x = np.linspace(0,2,1000)
#theta = wsin(d)
#eps = 0.1

#ns_fit = interpNS(data=x,mesh=d,k=2).dot(theta)
#mlp_fit = interpBPP(data=x,mesh=d,k=2).dot(theta)

#import matplotlib.pyplot as plt
#plt.plot(x,mlp_fit,'b-')
#plt.plot(x,ns_fit,'k--')
#plt.scatter(d,theta,c='r',s=30)
#plt.vlines(d,min(theta)-eps,max(theta)+eps,alpha=0.5)
#plt.show()



