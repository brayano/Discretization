import numpy as np
from scipy.spatial.distance import cdist

## Create functions for bivariate nearest neighbor and linear weighting

# Nearest Neighbor
# The targets, theta, will come in as a matrix.

def nearest1(x1,x2,mesh1,mesh2):
	n = x1.size	
	m = mesh1.size
	mids1 = mesh1[:-1]+.5*np.diff(mesh1)
	grid1 = np.sort(np.concatenate((mesh1,mids1)))
	mids2 = mesh2[:-1]+.5*np.diff(mesh2)
	grid2 = np.sort(np.concatenate((mesh2,mids2)))
	
	inds1a = np.digitize(x1,grid1)
	inds2a = np.digitize(x2,grid2)
	inds1b = np.floor(inds1a/2.0)
	inds2b = np.floor(inds2a/2.0)

	O = np.zeros((n,m*m))
	for i in range(n):
		working = np.zeros((m,m))
		working[inds1b[i],inds2b[i]] = 1.0
		O[i,] = np.ndarray.flatten(working)
	return O


def linear3(x1,x2,mesh1,mesh2):
	n = x1.size
	m = mesh1.size
	# Ensure the arrays are in column format for distance calculations
	x1 = x1.reshape((n,1))
	x2 = x2.reshape((n,1))
	X = np.concatenate((x1,x2),1)
	# Also, need to repeat the indivdual meshes to get the 2d-grid in a vector, the same order as Oh and theta reshaped in meshy2 solver.
	m1 = np.repeat(mesh1,m)
	m2 = np.tile(mesh2,m)
	m1 = m1.reshape((m*m,1))
	m2 = m2.reshape((m*m,1))
	Mesh = np.concatenate((m1,m2),1)

	dists = cdist(X,Mesh) # Default metric is Euclidean. In each row of dists, the Euclidean distance of X_1 from each coordinate of Mesh
	
	O = np.zeros((n,m*m))
	for i in range(n):
		working = np.zeros((m,m))
		zero_element = np.where(dists[i,]==0)[0] # those with distance 0 will get fully weighted at that point
		if zero_element.size == 0:
			near3ind = dists[i,].argsort()[:3]
			total_dist = np.sum(dists[i,near3ind])
			O[i,near3ind] = 1-dists[i,near3ind]/total_dist
		else:
			O[i,zero_element] = 1.0
	return O


def interpO(x1,x2,mesh1,mesh2,interp):
	# key=0: NS, key=1: BPP
	if interp == 0:
		O = nearest1(x1,x2,mesh1,mesh2)
	elif interp == 1:
		O = linear3(x1,x2,mesh1,mesh2)
	else:
		raise Exception("Not an option! Only interp=0 (nearest neighbor),1 (linear weighting)")
	return O


#d1 = np.linspace(0,1,10)
#x1 = np.random.sample(10)
#d2 = np.linspace(0,2,10)
#x2 = 2*np.random.sample(10)
#O = linear3(x1,x2,mesh1=d1,mesh2=d2)

#print O
