import utils
import discsolvers as ds
import numpy as np
import cvxpy as cvx
import scipy as scipy
import cvxopt as cvxopt
import matplotlib.pyplot as plt
import plotdisc as pld
import piece 

# Treating fitted values as constant within discretization.
# Number of cuts, C=100, and First difference matrix, k=1
# L-norm = 1
# Create random data for response
np.random.seed([117])
x = np.random.normal(0,2,200)
y = piece.piece1(x)+np.random.normal(0,1,200)

fig = plt.figure(figsize=(6, 4))
pld.dtplot(x,y,vtune=[1,10,30],ncuts=100,truefxn=piece.piece1,k=1,constant=True,lnorm=1)
fig.savefig("pieceC100K1constant.png")

###############################################
# Number of cuts, C=10, and 1st order diff., k=1
fig = plt.figure(figsize=(6, 4))
pld.dtplot(x,y,vtune=[1,10,30],ncuts=10,truefxn=piece.piece1,k=1,constant=True,lnorm=1)
fig.savefig("pieceC10K1constant.png")

###############################################
# Number of cuts, C=50, and L-norm, k=1
fig = plt.figure(figsize=(6, 4))
pld.dtplot(x,y,vtune=[1,10,30],ncuts=50,truefxn=piece.piece1,k=1,constant=True,lnorm=1)
fig.savefig("pieceC50K1constant.png")

###############################################
# # LINEAR INTERPOLATION OF DISCRETE EFFECTS
# Number of cuts, C=100, and 2nd-order diff., k=2
fig = plt.figure(figsize=(6, 4))
pld.dtplot(x,y,vtune=[1,10,30],ncuts=100,truefxn=piece.piece1,k=2,constant=False,lnorm=1)
fig.savefig("pieceC100K2linear.png")

# Number of cuts, C=50, and 2nd-order diff., k=2
fig = plt.figure(figsize=(6, 4))
pld.dtplot(x,y,vtune=[1,10,30],ncuts=50,truefxn=piece.piece1,k=2,constant=False,lnorm=1)
fig.savefig("pieceC50K2linear.png")

# Number of cuts, C=10, and 2nd-order diff., k=2
fig = plt.figure(figsize=(6, 4))
pld.dtplot(x,y,vtune=[1,10,30],ncuts=10,truefxn=piece.piece1,k=2,constant=False,lnorm=1)
fig.savefig("pieceC10K2linear.png")