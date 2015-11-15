import numpy as np
import cvxpy as cvx
import scipy as scipy
import cvxopt as cvxopt

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

def form_O(x,ncuts):
	segs = np.linspace(min(x),max(x),ncuts)
	cats = np.digitize(x, bins=segs)
	n = x.size
	O = np.zeros((n, ncuts))
	for i in range(n):
		O[i,cats[i]-1] = 1
	return O

def form_O_linear(x,ncuts):
	segs = np.linspace(min(x),max(x),ncuts)
	cats = np.digitize(x, bins=segs)
	t = x-segs[cats-1]
	delta = segs[1]
	n = x.size
	O = np.zeros((n, ncuts))
	for i in range(n):
		index = cats[i] - 1
		O[i,index] = t[i]/delta 
		if index+1 < ncuts:
			O[i,index+1] = (delta-t[i])/delta
	return O
	

def linterp(x,segs,theta):
	# x needs to be ordered: increasing.
	cats = np.digitize(x, bins=segs)
	t = x-segs[cats-1]
	delta = segs[1]
	shifttheta = np.delete(x,[0])
	shifttheta = np.append(x,0)
	len = x.size
	lin_fits = np.zeros(len)
	for i in range(len):
		lin_fits[i] = (t[i]*theta[cats[i]-1]+(delta-t[i])*shifttheta[cats[i]-1])/delta
	return lin_fits
	
	
