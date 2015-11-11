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
# form_o linear comes in as a flag to disc_one 

#np.random.seed([117])
#x = np.random.normal(1,5,200)
#y = 2*x**2-4*x-10+np.random.normal(0,1,200)

#print disc_one(x,y,tune=10,ncuts=10)