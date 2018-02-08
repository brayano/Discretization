import meshy as meshy
import meshy2 as meshy2
import numpy as np
import gendata_MeshySims as gen
import itertools 

# In this file, we create the simulation functions.
# For a generated data set, 
# we want to compare
# a configuration of meshy
# against meshy with a knot at every point.
# For some configs, that second setting will be equivalent to other problems. 

# For each data set, do we want to run multiple m's?

def univar_one(args):
	n = args[0]; ftrue = args[1]
	err_mu = args[2]; err_sig = args[3] 
	m = args[4] # either a list or a single value for the number of cuts
	# If given a list in "m", then this function will produce
	# a list with fitted values for each of the m's
	k = args[5]
	interp = args[6] # 1 is BPP, 0 is spline
	k_interp = args[7]
	lnorm = args[8]
	ntune = args[9] # number of tuning parameters to try: np.exp(np.linspace(-5,5,ntune))

	x_unordered = np.random.sample(n) # range between 0 and 1
	x = x_unordered[np.argsort(x_unordered)]
	y = gen.gen_univar(x=x, fxn=ftrue, error_mu=err_mu, error_sig=err_sig)
	ytrue = ftrue(x)
	#trendy_modfits =meshy.meshy(x, y, m=n, k=k, interp=interp, mesh=x, lnorm =lnorm, ntune=ntune)['minmse.fits']['fitted']
	
	len_m = len(m)
	#meshy_fits = []
	meshy_mses = []
	#meshy_lams=[]
	#approx_error = np.zeros(len_m)
	for i in range(len_m):
		meshy_mod = meshy.meshy(x,y,m=m[i], k=k, interp=interp, ftrue=ytrue, k_interp=k_interp,lnorm = lnorm, ntune=ntune) 
		#meshy_fits.append(meshy_mod['minmse.fits']['fitted'])
		meshy_mses.append(meshy_mod['minmse'])
		#meshy_lams.append(meshy_mod['minmse.lam'])
		#approx_error[i] = np.mean(meshy_fits[i]-trendy_modfits)
	
	#approx_error = np.mean(meshy_fits-trendy_mod['minmse.fits'],axis=1) # This of length len_m
	#output = {'m': m, 'approx_error': approx_error,'stat_error': meshy_mses}
	#output = np.concatenate((meshy_mses,approx_error))
	output = np.array(meshy_mses)
	#output = np.concatenate((meshy_mses,meshy_lams))
	return output

def bivar_one(args):
	n = args[0]; ftrue = args[1]
	err_mu = args[2]; err_sig = args[3] 
	m = args[4] # either a list or a single value for the number of cuts
	# If given a list in "m", then this function will produce
	# a list with fitted values for each of the m's
	k1 = args[5]
	k2 = args[6]
	interp = args[7] # 0 is nearest neighbor, 1 is linear
	lnorm = args[8]
	ntune = args[9] # number of tuning parameters to try: np.exp(np.linspace(-5,5,ntune))

	x1_unordered = np.random.sample(n) # range between 0 and 1
	x1 = x1_unordered[np.argsort(x1_unordered)]
	x2 = np.random.sample(n)
	dargs = [x1,x2]
	y = gen.gen_bivar(x1=x1, x2=x2, fxn=ftrue, error_mu=err_mu, error_sig=err_sig)
	ytrue = ftrue(dargs)
	#trendy_modfits =meshy.meshy(x, y, m=n, k=k, interp=interp, mesh=x, lnorm =lnorm, ntune=ntune)['minmse.fits']['fitted']
	
	len_m = len(m)
	#meshy_fits = []
	meshy_mses = []
	#approx_error = np.zeros(len_m)
	for i in range(len_m):
		meshy_mod = meshy2.meshy2(x1,x2,y,m=m[i], ftrue=ytrue, k1=k1,k2=k2, interp=interp, lnorm = lnorm, ntune=ntune) 
		#meshy_fits.append(meshy_mod['minmse.fits']['fitted'])
		meshy_mses.append(meshy_mod['minmse'])
		#approx_error[i] = np.mean(meshy_fits[i]-trendy_modfits)
	
	#approx_error = np.mean(meshy_fits-trendy_mod['minmse.fits'],axis=1) # This of length len_m
	#output = {'m': m, 'approx_error': approx_error,'stat_error': meshy_mses}
	#output = np.concatenate((meshy_mses,approx_error))
	output = np.array(meshy_mses)
	return output

def bivar_two(args):
	# Specifically for sombrero function
	n = args[0]; ftrue = args[1]
	err_mu = args[2]; err_sig = args[3] 
	m = args[4] # either a list or a single value for the number of cuts
	# If given a list in "m", then this function will produce
	# a list with fitted values for each of the m's
	k1 = args[5]
	k2 = args[6]
	interp = args[7] # 0 is nearest neighbor, 1 is linear
	lnorm = args[8]
	ntune = args[9] # number of tuning parameters to try: np.exp(np.linspace(-5,5,ntune))

	x1_unordered = np.random.uniform(-1,1,n) # range between -1 and 1
	x1 = x1_unordered[np.argsort(x1_unordered)]
	x2 = np.random.uniform(-1,1,n)
	dargs = [x1,x2]
	y = gen.gen_bivar(x1=x1, x2=x2, fxn=ftrue, error_mu=err_mu, error_sig=err_sig)
	ytrue = ftrue(dargs)
	#trendy_modfits =meshy.meshy(x, y, m=n, k=k, interp=interp, mesh=x, lnorm =lnorm, ntune=ntune)['minmse.fits']['fitted']
	
	len_m = len(m)
	#meshy_fits = []
	meshy_mses = []
	#approx_error = np.zeros(len_m)
	for i in range(len_m):
		meshy_mod = meshy2.meshy2(x1,x2,y,m=m[i], ftrue=ytrue, k1=k1,k2=k2, interp=interp, lnorm = lnorm, ntune=ntune) 
		#meshy_fits.append(meshy_mod['minmse.fits']['fitted'])
		meshy_mses.append(meshy_mod['minmse'])
		#approx_error[i] = np.mean(meshy_fits[i]-trendy_modfits)
	
	#approx_error = np.mean(meshy_fits-trendy_mod['minmse.fits'],axis=1) # This of length len_m
	#output = {'m': m, 'approx_error': approx_error,'stat_error': meshy_mses}
	#output = np.concatenate((meshy_mses,approx_error))
	output = np.array(meshy_mses)
	return output

#np.random.seed(117)
#x = np.random.sample(100)
#x = x[np.argsort(x)]
#print np.unique(x).size
#print x[0],x[99]
#meshbins = np.diff(x)
#print min(meshbins), max(meshbins)
#y = gen.gen_univar(x=x,fxn=gen.piece_linear, error_mu=0,error_sig=0.1)
#allknots = meshy.meshy(x,y,m=100,k=1,mesh=x,interp=1,lnorm=1,ntune=100)

#print allknots['minmse'], allknots['minmse.lam']
