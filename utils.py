import numpy as np
import cvxpy as cvx
import scipy as scipy
import cvxopt as cvxopt
import matplotlib.pyplot as plt

def form_D1(n):
	D = np.zeros((n-1,n))
	for i in range(n-1):
		D[i,i] = 1
		D[i,i+1] = -1
	return D 

def form_Dk(n,k):
	D = np.identity(n)
	for i in range(k):
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
	inds = np.digitize(x,mesh)-1
	psitilde = designBPP(x=data,k=k)
	O = np.zeros((data.size,mesh.size))
	for i in range(data.size):
		Psi = designBPP(x=mesh[np.arange(inds[i],inds[i]+k+1)],k=k)
		if inds[i] >= mesh.size-1-k:
			O[i,(mesh.size-1-k):(mesh.size)] = psitilde[i,].dot(np.linalg.inv(Psi))
		else:
			O[i,inds[i]:(inds[i]+k+1)] =  psitilde[i,].dot(np.linalg.inv(Psi))
	return O


#d = np.linspace(0,1,10)
#x = np.random.sample(5)
#print x
#Psi = designNS(x=d,mesh=d,k=3)
#psitilde = designNS(x=x,mesh=d,k=3)
#print psitilde.dot(np.linalg.inv(Psi))
#Psi = designBPP(x=d[np.arange(1,3)],k=1)
#print d[np.arange(1,2)]
#print designBPP(x,k=1)[1,].dot(np.linalg.inv(Psi))
#print interpNS(data=x,mesh=d,k=0)
#print interpBPP(data=x,mesh=d,k=0)
