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

def form_O(x,ncuts,eps=0.01):
	segs = np.linspace(min(x)-eps,max(x)+eps,ncuts)
	# Default: bins[i-1]<= x < bins[i]
	cats = np.digitize(x, bins=segs)
	n = x.size
	O = np.zeros((n, ncuts))
	for i in range(n):
		O[i,cats[i]-1] = 1
	return O

def form_O_linear(x,ncuts,eps=0.01):
	# Produce a linear interpolation knot matrix. 
	# Unless segments are provided, we assume equal bin
	# sizes. 
	segs = np.linspace(min(x)-eps,max(x)+eps,ncuts)
	cats = np.digitize(x, bins=segs)
	delta = segs[1]-segs[0] # length of each bin
	
	# Calculate the distance between a point and the beginning
	# of its corresponding bin. 
	t = x-segs[cats-1] 
	n = x.size
	O = np.zeros((n, ncuts))
	for i in range(n):
		index = cats[i] - 1
		O[i,index] = t[i]/delta 
		if index+1 < ncuts:
			O[i,index+1] = (delta-t[i])/delta
	return O

def Omatrix(x,ncuts,constant=True,eps=0.01):
	# Currently, Omatrix outputs either constant knot matrix
	# or linearly interpolated knot matrix. Eventually,
	# "constant" will be deprecated for "degree" option.
	# degree=0; constant. degree=1; linear. degree=2; quadratic. etc. 
	if constant == True:
		O = form_O(x,ncuts,eps=eps)
	if constant != True:
		O = form_O_linear(x,ncuts,eps=eps)
	return O

