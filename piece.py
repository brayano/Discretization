import numpy as np

def piece1(x):
	n = x.size
	y = np.zeros(n)
	for i in range(n):
		if x[i] < 0:
			y[i] = .5*x[i]**2+3*x[i]
		if x[i] >= 0:
			y[i] = np.exp(-x[i])*np.sin(10*x[i])
			
	return y
	
def piece2(x):
	n = x.size
	y = np.zeros(n)
	for i in range(n):
		if x[i] < 0:
			y[i] = -np.sin(2*x[i])
		if 0 <= x[i] <= 2:
			y[i] = 3
		if x[i]>2:
			y[i] = np.exp(-x[i])
	return y

